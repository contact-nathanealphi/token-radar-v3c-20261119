// Token Radar, variant C: launch-odds-weighted comps ("odds-weighted-1").
//
// One sealed snapshot in, one ranked board out. Every number below is fitted
// at run time from the snapshot's own 758 protocols; nothing is hardcoded
// by protocol name and no outside data is read.
//
// The model, in five steps:
//
//   1. Peer multiple. For each category, the geometric mean of the
//      market-cap/TVL multiples of its traded protocols (multiples
//      winsorized to the global 5th/95th percentiles), shrunk toward the
//      global geometric mean with 20 pseudo-peers, clamped to [0.02, 2.5].
//      Categories with no traded peer fall back to the global mean.
//   2. Size tilt. exp(slope * (log10(TVL) - mean log10(TVL))), where slope
//      is the OLS slope of log(multiple / category multiple) on centred
//      log10(TVL) among traded protocols, clipped to [-1.5, 0]. Big
//      protocols carry lower multiples in this snapshot, and the tilt
//      says so with a fitted number instead of hiding it.
//   3. Growth tilt. exp(beta * clipped 7-day growth), where beta is the OLS
//      slope of log(multiple) on growth among traded protocols with a
//      market cap, clipped to [0, 0.01] per point. In this snapshot the
//      fitted slope is not positive, so the tilt is 1.0 and the dashboard
//      reports exactly that: growth carries no positive premium here.
//   4. Seasoning. 0.85 + 0.15 * min(age_days, 1095) / 1095; unknown age
//      takes the midpoint. A bounded prior, stated as one.
//   5. Launch odds. A logistic regression fitted on all 758 protocols:
//      y = 1 when the protocol records a token, 0 otherwise. Features are
//      log TVL, log age, clipped 7-day growth, chain count and the
//      category's shrunk traded rate. This is a cross-sectional fit, NOT a
//      prediction that or when a token is issued: it scores how much a
//      protocol resembles the ones that already record a token.
//
//   raw value      = TVL * peer multiple * size tilt * growth tilt * seasoning
//   expected value = raw value * launch odds
//
// The board ranks by expected value. implied_value_usd is the expected
// value; the dashboard shows the raw value, the odds and the interval
// beside it, so every rank can be recomputed from the snapshot.
//
// Validation is computed at run time and reported honestly, warts and all:
//   - Odds model: 5-fold cross-validation with folds from a slug hash.
//     Category rates and standardisation are refit inside each training
//     fold, so no held-out row leaks into its own prediction. Metrics:
//     AUC, Brier score, log loss, accuracy at 0.5, against the
//     predict-the-global-rate baseline.
//   - Value model: leave-one-out over traded protocols with a market cap.
//     Each protocol's multiple, size slope and growth slope come from the
//     other protocols only, never itself. Metrics: median absolute log
//     error and share within 2x, against the flat global-median baseline.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;
const MODEL_VERSION = 'odds-weighted-1';

/** Model constants, all stated in the dashboard's methodology panel. */
export const MODEL = {
  version: MODEL_VERSION,
  peer_prior: 20,
  multiple_floor: 0.02,
  multiple_cap: 2.5,
  growth_floor: -60,
  growth_cap: 120,
  beta_floor: 0,
  beta_cap: 0.01,
  size_slope_floor: -1.5,
  size_slope_cap: 0,
  seasoning_floor: 0.85,
  seasoning_range: 0.15,
  seasoning_days: 1095,
  odds_floor: 0.02,
  odds_cap: 0.98,
  category_prior: 10,
  gd_iterations: 2000,
  gd_rate: 0.1,
  gd_l2: 1.0,
  folds: 5,
  verdict_farm_odds: 0.45,
  verdict_farm_value: 15_000_000,
  verdict_thin_odds: 0.3,
  verdict_thin_value: 3_000_000,
  live_rows: 40,
} as const;

// ------------------------------------------------------------------ helpers

const finite = (v: unknown, fb: number): number => (typeof v === 'number' && Number.isFinite(v) ? v : fb);
const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

function quantile(xs: number[], q: number): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const i = clamp(q * (s.length - 1), 0, s.length - 1);
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return s[lo]! + (s[hi]! - s[lo]!) * (i - lo);
}

/** FNV-1a hash, for deterministic cross-validation folds. */
function hashSlug(slug: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < slug.length; i++) {
    h ^= slug.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Whole days from `listedAt` to `asOf`, clamped at zero; null when absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** An ISO date `offset` days after `asOf`. */
export function dateAfter(asOf: string, offset: number): string {
  const t = Date.parse(`${asOf}T00:00:00Z`) + offset * DAY_MS;
  return new Date(t).toISOString().slice(0, 10);
}

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
/** A whole word wrapped, staked or staking, in any case. */
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;
/** Exchange names whose prefix marks a product of an exchange. */
const EXCHANGES = ['Binance', 'Coinbase', 'Bybit', 'Kraken', 'Bitget', 'OKX', 'Bitfinex', 'Gemini'];
/** Categories that mark a wrapped, staked or bridged product, or an exchange itself. */
const EXCLUDED_CATEGORIES = new Set(['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX']);

/** Why a pre-token protocol is left off the ranked board, or null when rankable. */
export function exclusionReason(p: SnapshotProtocol): string | null {
  if (p.category !== null && EXCLUDED_CATEGORIES.has(p.category)) return 'category';
  if (VERSIONED.test(p.name)) return 'versioned';
  if (WRAPPED_WORD.test(p.name)) return 'wrapped-word';
  if (EXCHANGES.some((e) => p.name.startsWith(e))) return 'exchange-prefix';
  return null;
}

/** Whether a protocol is one the board may rank: pre-token and not an excluded product. */
export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && exclusionReason(p) === null;
}

// ------------------------------------------------------------ value fitting

/** Everything the value model fits from a protocol set. */
export interface ValueFit {
  /** Winsor bounds for multiples, from the global 5th/95th percentiles. */
  winLo: number;
  winHi: number;
  /** Global geometric mean of winsorized multiples. */
  globalGeo: number;
  /** Global median of winsorized multiples (the validation baseline). */
  globalMedian: number;
  /** Per-category peer multiples: log-sum, count, and winsorized quartiles. */
  byCategory: Map<string, { n: number; logSum: number; p25: number; p75: number }>;
  globalP25: number;
  globalP75: number;
  /** OLS slope of log(multiple / category multiple) on centred log10(TVL). */
  sizeSlope: number;
  /** Mean log10(TVL) among traded protocols with a market cap. */
  meanLogTvl: number;
  /** OLS slope of log(multiple) on clipped growth, clipped to [0, 0.01]. */
  betaGrowth: number;
  tradedWithMcap: number;
}

function tradedWithMcap(protocols: SnapshotProtocol[]): SnapshotProtocol[] {
  return protocols.filter((p) => !isPreToken(p) && p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
}

/** Fit the value model on a protocol set: peer multiples, size slope, growth slope. */
export function fitValue(protocols: SnapshotProtocol[]): ValueFit {
  const traded = tradedWithMcap(protocols);
  const multiples = traded.map((p) => p.mcap_usd! / p.tvl_usd);
  const winLo = quantile(multiples, 0.05);
  const winHi = quantile(multiples, 0.95);
  const win = (m: number): number => clamp(m, winLo, winHi);
  const winsorized = multiples.map(win);
  const globalGeo = Math.exp(winsorized.reduce((s, m) => s + Math.log(m), 0) / Math.max(1, winsorized.length));
  const globalMedian = median(winsorized);
  const globalP25 = quantile(winsorized, 0.25);
  const globalP75 = quantile(winsorized, 0.75);
  const groups = new Map<string, number[]>();
  for (const p of traded) {
    const key = p.category ?? '(none)';
    const arr = groups.get(key) ?? [];
    arr.push(win(p.mcap_usd! / p.tvl_usd));
    groups.set(key, arr);
  }
  const byCategory = new Map<string, { n: number; logSum: number; p25: number; p75: number }>();
  for (const [key, arr] of groups) {
    byCategory.set(key, { n: arr.length, logSum: arr.reduce((s, m) => s + Math.log(m), 0), p25: quantile(arr, 0.25), p75: quantile(arr, 0.75) });
  }
  const prior = MODEL.peer_prior;
  const logGlobal = Math.log(globalGeo);
  const catGeo = (category: string | null): number => {
    const e = byCategory.get(category ?? '(none)');
    if (e === undefined) return globalGeo;
    return Math.exp((e.logSum + prior * logGlobal) / (e.n + prior));
  };
  // Size slope: how log(multiple / category multiple) moves with centred log10(TVL).
  const meanLogTvl = traded.reduce((s, p) => s + Math.log10(p.tvl_usd), 0) / Math.max(1, traded.length);
  let num = 0;
  let den = 0;
  for (const p of traded) {
    const x = Math.log10(p.tvl_usd) - meanLogTvl;
    const y = Math.log(win(p.mcap_usd! / p.tvl_usd) / catGeo(p.category));
    num += x * y;
    den += x * x;
  }
  const sizeSlope = den === 0 ? 0 : clamp(num / den, MODEL.size_slope_floor, MODEL.size_slope_cap);
  // Growth slope: univariate OLS of log(multiple) on clipped growth.
  const pts = traded.map((p) => [clamp(finite(p.change_7d, 0), MODEL.growth_floor, MODEL.growth_cap), Math.log(win(p.mcap_usd! / p.tvl_usd))] as const);
  let betaGrowth = 0;
  if (pts.length >= 10) {
    const mx = pts.reduce((s, pt) => s + pt[0], 0) / pts.length;
    const my = pts.reduce((s, pt) => s + pt[1], 0) / pts.length;
    let n2 = 0;
    let d2 = 0;
    for (const [x, y] of pts) {
      n2 += (x - mx) * (y - my);
      d2 += (x - mx) * (x - mx);
    }
    betaGrowth = d2 === 0 ? 0 : clamp(n2 / d2, MODEL.beta_floor, MODEL.beta_cap);
  }
  return { winLo, winHi, globalGeo, globalMedian, byCategory, globalP25, globalP75, sizeSlope, meanLogTvl, betaGrowth, tradedWithMcap: traded.length };
}

/** A category's shrunk geometric-mean multiple, with shrunk quartile bounds. */
export function categoryMultiple(category: string | null, fit: ValueFit): { multiple: number; peers: number; low: number; high: number } {
  const prior = MODEL.peer_prior;
  const logGlobal = Math.log(fit.globalGeo);
  const e = fit.byCategory.get(category ?? '(none)');
  if (e === undefined) return { multiple: fit.globalGeo, peers: 0, low: fit.globalGeo, high: fit.globalGeo };
  const multiple = clamp(Math.exp((e.logSum + prior * logGlobal) / (e.n + prior)), MODEL.multiple_floor, MODEL.multiple_cap);
  const low = clamp(
    Math.exp((e.n * Math.log(Math.max(e.p25, 1e-9)) + prior * Math.log(Math.max(fit.globalP25, 1e-9))) / (e.n + prior)),
    MODEL.multiple_floor,
    MODEL.multiple_cap,
  );
  const high = clamp(
    Math.exp((e.n * Math.log(Math.max(e.p75, 1e-9)) + prior * Math.log(Math.max(fit.globalP75, 1e-9))) / (e.n + prior)),
    MODEL.multiple_floor,
    MODEL.multiple_cap,
  );
  return { multiple, peers: e.n, low, high };
}

/** The three tilts and the raw value one protocol gets under a fitted value model. */
export function rawValue(
  p: SnapshotProtocol,
  asOf: string,
  fit: ValueFit,
): { raw: number; multiple: number; peers: number; low: number; high: number; sizeMult: number; growthMult: number; seasonMult: number; growthPct: number; age: number | null } {
  const age = ageDays(asOf, p.listed_at);
  const growthPct = clamp(finite(p.change_7d, 0), MODEL.growth_floor, MODEL.growth_cap);
  const { multiple, peers, low, high } = categoryMultiple(p.category, fit);
  const sizeMult = Math.exp(fit.sizeSlope * (Math.log10(Math.max(1, p.tvl_usd)) - fit.meanLogTvl));
  const growthMult = Math.exp(fit.betaGrowth * growthPct);
  const seasonMult = MODEL.seasoning_floor + (MODEL.seasoning_range * Math.min(age ?? 365, MODEL.seasoning_days)) / MODEL.seasoning_days;
  const raw = p.tvl_usd * multiple * sizeMult * growthMult * seasonMult;
  return { raw, multiple, peers, low, high, sizeMult, growthMult, seasonMult, growthPct, age };
}

// -------------------------------------------------------------- launch odds

interface OddsFit {
  weights: number[];
  means: number[];
  sds: number[];
  globalRate: number;
}

/** Raw features for the odds model, before standardisation. */
function rawFeatures(
  p: SnapshotProtocol,
  age: number | null,
  medianAge: number,
  categoryRate: (category: string | null) => number,
): number[] {
  const growth = clamp(finite(p.change_7d, 0), MODEL.growth_floor, MODEL.growth_cap);
  return [
    Math.log10(Math.max(1, p.tvl_usd)),
    Math.log1p(age ?? medianAge),
    growth / 60,
    Math.log1p(p.chains.length),
    categoryRate(p.category),
  ];
}

const sigmoid = (z: number): number => 1 / (1 + Math.exp(-clamp(z, -30, 30)));

/** Category traded rates shrunk toward the global rate, from a protocol set. */
function categoryRates(protocols: SnapshotProtocol[]): { rate: (category: string | null) => number; global: number } {
  const total = protocols.length;
  const traded = protocols.filter((p) => !isPreToken(p)).length;
  const global = total === 0 ? 0.5 : traded / total;
  const counts = new Map<string, { n: number; t: number }>();
  for (const p of protocols) {
    const key = p.category ?? '(none)';
    const e = counts.get(key) ?? { n: 0, t: 0 };
    e.n += 1;
    if (!isPreToken(p)) e.t += 1;
    counts.set(key, e);
  }
  const prior = MODEL.category_prior;
  return {
    global,
    rate: (category: string | null): number => {
      const e = counts.get(category ?? '(none)');
      if (e === undefined) return global;
      return (e.t + prior * global) / (e.n + prior);
    },
  };
}

/** Fit the logistic odds model on a protocol set by deterministic gradient descent. */
function fitOdds(protocols: SnapshotProtocol[], asOf: string): OddsFit {
  const ages = protocols.map((p) => ageDays(asOf, p.listed_at)).filter((a): a is number => a !== null);
  const medianAge = ages.length > 0 ? median(ages) : 365;
  const { rate, global } = categoryRates(protocols);
  const rows = protocols.map((p) => ({
    z: rawFeatures(p, ageDays(asOf, p.listed_at), medianAge, rate),
    y: isPreToken(p) ? 0 : 1,
  }));
  const k = rows[0]!.z.length;
  const means = Array.from({ length: k }, (_, j) => rows.reduce((s, r) => s + r.z[j]!, 0) / rows.length);
  const sds = Array.from({ length: k }, (_, j) => {
    const v = rows.reduce((s, r) => s + (r.z[j]! - means[j]!) ** 2, 0) / rows.length;
    return Math.sqrt(v) || 1;
  });
  const feats = rows.map((r) => [1, ...r.z.map((v, j) => (v - means[j]!) / sds[j]!)]);
  const w = new Array<number>(k + 1).fill(0);
  const n = feats.length;
  const lr = MODEL.gd_rate;
  const l2 = MODEL.gd_l2;
  for (let iter = 0; iter < MODEL.gd_iterations; iter++) {
    const grad = new Array<number>(k + 1).fill(0);
    for (let i = 0; i < n; i++) {
      const f = feats[i]!;
      let dot = 0;
      for (let j = 0; j <= k; j++) dot += w[j]! * f[j]!;
      const err = sigmoid(dot) - rows[i]!.y;
      for (let j = 0; j <= k; j++) grad[j]! += err * f[j]!;
    }
    for (let j = 0; j <= k; j++) {
      const reg = j === 0 ? 0 : (l2 * w[j]!) / n;
      w[j]! -= lr * (grad[j]! / n + reg);
    }
  }
  return { weights: w, means, sds, globalRate: global };
}

/** A protocol's launch odds under a fitted model. */
function launchOdds(p: SnapshotProtocol, asOf: string, fit: OddsFit, medianAge: number, rate: (c: string | null) => number): number {
  const z = rawFeatures(p, ageDays(asOf, p.listed_at), medianAge, rate);
  let dot = fit.weights[0]!;
  for (let j = 0; j < z.length; j++) dot += fit.weights[j + 1]! * ((z[j]! - fit.means[j]!) / fit.sds[j]!);
  return clamp(sigmoid(dot), MODEL.odds_floor, MODEL.odds_cap);
}

// ---------------------------------------------------------------- validation

export interface OddsValidation {
  auc: number;
  brier: number;
  logloss: number;
  accuracy: number;
  baseline_brier: number;
  baseline_logloss: number;
  folds: number;
  n: number;
}

export interface ValueValidation {
  median_abs_log_error: number;
  within_2x: number;
  baseline_median_abs_log_error: number;
  baseline_within_2x: number;
  n: number;
}

function aucScore(pairs: Array<[number, number]>): number {
  const pos = pairs.filter(([, y]) => y === 1).map(([s]) => s);
  const neg = pairs.filter(([, y]) => y === 0).map(([s]) => s);
  if (pos.length === 0 || neg.length === 0) return 0.5;
  let wins = 0;
  for (const a of pos) for (const b of neg) wins += a > b ? 1 : a === b ? 0.5 : 0;
  return wins / (pos.length * neg.length);
}

/** 5-fold cross-validation of the odds model; folds from a slug hash, refit per fold. */
export function validateOdds(snapshot: Snapshot): OddsValidation {
  const protocols = snapshot.protocols;
  const ages = protocols.map((p) => ageDays(snapshot.as_of, p.listed_at)).filter((a): a is number => a !== null);
  const medianAge = ages.length > 0 ? median(ages) : 365;
  const preds: Array<[number, number]> = [];
  for (let fold = 0; fold < MODEL.folds; fold++) {
    const train = protocols.filter((p) => hashSlug(p.slug) % MODEL.folds !== fold);
    const test = protocols.filter((p) => hashSlug(p.slug) % MODEL.folds === fold);
    const fit = fitOdds(train, snapshot.as_of);
    const { rate } = categoryRates(train);
    for (const p of test) {
      preds.push([launchOdds(p, snapshot.as_of, fit, medianAge, rate), isPreToken(p) ? 0 : 1]);
    }
  }
  const n = preds.length;
  const global = preds.reduce((s, [, y]) => s + y, 0) / n;
  let brier = 0;
  let logloss = 0;
  let correct = 0;
  for (const [s, y] of preds) {
    brier += (s - y) ** 2;
    logloss += -(y * Math.log(clamp(s, 1e-9, 1 - 1e-9)) + (1 - y) * Math.log(clamp(1 - s, 1e-9, 1 - 1e-9)));
    correct += (s >= 0.5 ? 1 : 0) === y ? 1 : 0;
  }
  const baselineBrier = global * (1 - global) ** 2 + (1 - global) * global ** 2;
  const baselineLogloss = -(global * Math.log(clamp(global, 1e-9, 1 - 1e-9)) + (1 - global) * Math.log(clamp(1 - global, 1e-9, 1 - 1e-9)));
  return {
    auc: round4(aucScore(preds)),
    brier: round4(brier / n),
    logloss: round4(logloss / n),
    accuracy: round4(correct / n),
    baseline_brier: round4(baselineBrier),
    baseline_logloss: round4(baselineLogloss),
    folds: MODEL.folds,
    n,
  };
}

/** Leave-one-out validation of the value model: every fit excludes the row it scores. */
export function validateValue(snapshot: Snapshot): ValueValidation {
  const traded = tradedWithMcap(snapshot.protocols);
  const full = fitValue(snapshot.protocols);
  const errs: number[] = [];
  const baseErrs: number[] = [];
  for (const p of traded) {
    const others = snapshot.protocols.filter((q) => q.slug !== p.slug);
    const fit = fitValue(others);
    const v = rawValue(p, snapshot.as_of, fit);
    errs.push(Math.abs(Math.log(v.raw / p.mcap_usd!)));
    baseErrs.push(Math.abs(Math.log((p.tvl_usd * full.globalMedian) / p.mcap_usd!)));
  }
  const within = (es: number[]): number => es.filter((e) => e <= Math.log(2)).length / Math.max(1, es.length);
  return {
    median_abs_log_error: round4(median(errs)),
    within_2x: round4(within(errs)),
    baseline_median_abs_log_error: round4(median(baseErrs)),
    baseline_within_2x: round4(within(baseErrs)),
    n: traded.length,
  };
}

// --------------------------------------------------------------------- board

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  model: {
    peer_multiple: number;
    peers: number;
    size_multiplier: number;
    growth_pct: number;
    growth_multiplier: number;
    seasoning_multiplier: number;
    raw_value_usd: number;
    launch_odds: number;
    implied_low_usd: number;
    implied_high_usd: number;
    confidence: 'high' | 'medium' | 'low';
    verdict: 'farm' | 'thin' | 'watch';
    flags: string[];
  };
  farm_steps: string[];
}

/** The dashboard's compact view of one row. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  raw_value_usd: number;
  launch_odds: number;
  implied_low_usd: number;
  implied_high_usd: number;
  peer_multiple: number;
  size_multiplier: number;
  growth_multiplier: number;
  seasoning_multiplier: number;
  peers: number;
  confidence: 'high' | 'medium' | 'low';
  verdict: 'farm' | 'thin' | 'watch';
  watchlist: boolean;
  flags: string[];
  farm_steps: string[];
}

/** A live-token protocol: it already trades, so the board shows the market's own value. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
  multiple: number | null;
}

/** Full accounting of the snapshot: every protocol counted, none dropped silently. */
export interface Accounting {
  total: number;
  traded: number;
  traded_with_mcap: number;
  pre_token: number;
  ranked: number;
  watchlist: number;
  early: number;
  excluded_category: number;
  excluded_versioned: number;
  excluded_wrapped_word: number;
  excluded_exchange_prefix: number;
}

/** Everything the dashboard renders. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  growth_beta: number;
  size_slope: number;
  mean_log_tvl: number;
  global_multiple: number;
  global_traded_rate: number;
  odds_validation: OddsValidation;
  value_validation: ValueValidation;
  accounting: Accounting;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  verdict_rule: string;
  rows: RadarRow[];
  live: LiveRow[];
}

/** The verdict from the numbers alone: odds and expected value against fixed thresholds. */
export function verdictFor(odds: number, expected: number): 'farm' | 'thin' | 'watch' {
  if (odds >= MODEL.verdict_farm_odds && expected >= MODEL.verdict_farm_value) return 'farm';
  if (odds >= MODEL.verdict_thin_odds && expected >= MODEL.verdict_thin_value) return 'thin';
  return 'watch';
}

/** Confidence from peer support: how many traded peers back the category multiple. */
function confidenceFor(peers: number): 'high' | 'medium' | 'low' {
  if (peers >= 8) return 'high';
  if (peers >= 3) return 'medium';
  return 'low';
}

/** The protocols the board ranks, strongest first, with ranks 1..n. */
export function buildBoard(snapshot: Snapshot): { board: BoardRow[]; valueFit: ValueFit } {
  const valueFit = fitValue(snapshot.protocols);
  const oddsFit = fitOdds(snapshot.protocols, snapshot.as_of);
  const ages = snapshot.protocols.map((p) => ageDays(snapshot.as_of, p.listed_at)).filter((a): a is number => a !== null);
  const medianAge = ages.length > 0 ? median(ages) : 365;
  const { rate } = categoryRates(snapshot.protocols);

  const chosen = new Map<string, SnapshotProtocol>();
  for (const slug of snapshot.watchlist) {
    const p = snapshot.protocols.find((q) => q.slug === slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  for (const p of snapshot.protocols) {
    if (!chosen.has(p.slug) && isRankable(p)) chosen.set(p.slug, p);
  }

  const scored = [...chosen.values()].map((p) => {
    const v = rawValue(p, snapshot.as_of, valueFit);
    const odds = launchOdds(p, snapshot.as_of, oddsFit, medianAge, rate);
    const expected = Math.max(1, Math.round(v.raw * odds));
    const flags: string[] = [];
    if (v.peers === 0) flags.push('no traded peer in category; global mean used');
    else if (v.peers < 3) flags.push(`only ${v.peers} traded peer${v.peers === 1 ? '' : 's'} in category`);
    if (p.change_7d === null) flags.push('7d growth not reported; treated as 0%');
    if (v.age === null) flags.push('listing date not reported; median age used');
    return { p, v, odds, expected, flags };
  });
  scored.sort((a, b) => b.expected - a.expected || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));

  const board: BoardRow[] = scored.map((s, i) => {
    const tilt = s.v.sizeMult * s.v.growthMult * s.v.seasonMult;
    const lowUsd = Math.max(1, Math.round(s.p.tvl_usd * s.v.low * tilt * s.odds));
    const highUsd = Math.max(s.expected, Math.round(s.p.tvl_usd * s.v.high * tilt * s.odds));
    return {
      rank: i + 1,
      slug: s.p.slug,
      name: s.p.name,
      category: s.p.category,
      chains: s.p.chains,
      tvl_usd: s.p.tvl_usd,
      change_7d: s.p.change_7d,
      mcap_usd: s.p.mcap_usd,
      listed_at: s.p.listed_at,
      gecko_id: s.p.gecko_id,
      symbol: s.p.symbol,
      token_status: 'pre-token',
      age_days: s.v.age,
      implied_value_usd: s.expected,
      model: {
        peer_multiple: round4(s.v.multiple),
        peers: s.v.peers,
        size_multiplier: round4(s.v.sizeMult),
        growth_pct: s.v.growthPct,
        growth_multiplier: round4(s.v.growthMult),
        seasoning_multiplier: round4(s.v.seasonMult),
        raw_value_usd: Math.max(1, Math.round(s.v.raw)),
        launch_odds: round4(s.odds),
        implied_low_usd: lowUsd,
        implied_high_usd: highUsd,
        confidence: confidenceFor(s.v.peers),
        verdict: verdictFor(s.odds, s.expected),
        flags: s.flags,
      },
      farm_steps: farmSteps(s.p, s.v.age, s.odds, s.expected, snapshot.as_of),
    };
  });
  return { board, valueFit };
}

// ---------------------------------------------------------------- farm plans

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 45;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/**
 * Planning gas estimates by chain family. These are NOT snapshot data: the
 * snapshot carries no fees. They are coarse planning figures so a reader can
 * budget, labelled "est." wherever they appear.
 */
const CHEAP_CHAINS = new Set([
  'Base', 'Arbitrum', 'Optimism', 'Polygon', 'BSC', 'Avalanche', 'Solana', 'Sui', 'Aptos',
  'zkSync Era', 'Linea', 'Blast', 'Mantle', 'Scroll', 'Berachain', 'Unichain', 'Sei',
  'Gnosis', 'Fantom', 'Celo', 'Moonbeam', 'Kaia', 'Story', 'MegaETH', 'Hyperliquid',
  'Arbitrum One', 'OP Mainnet', 'BNB Chain', 'Tron', 'Near', 'Starknet', 'Abstract',
]);

function gasTier(chain: string): { label: string; weekly: string } {
  if (chain === 'Ethereum') return { label: 'est. $8–$25 of gas', weekly: 'est. $5–$15 of gas a week' };
  if (CHEAP_CHAINS.has(chain)) return { label: 'est. $1–$3 of gas', weekly: 'est. under $1 of gas a week' };
  return { label: 'est. $3–$8 of gas', weekly: 'est. $2–$5 of gas a week' };
}

/** The category's one action, in words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} of USDC to ${name} on ${chain} and keep the position open`;
  if (c.includes('dex')) return `Provide $${capital} of liquidity to ${name}'s main pool on ${chain} and keep the position open`;
  if (c.includes('yield') || c.includes('leveraged farm')) return `Deposit $${capital} into ${name}'s highest-TVL vault on ${chain} and keep the shares`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC to ${chain} through ${name} and use the bridged funds in its app`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${capital} of the chain's staking asset through ${name} and keep the receipt token`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Deposit $${capital} of USDC as margin on ${name} and open one small hedged position`;
  if (c.includes('prediction')) return `Fund $${capital} and place one market position on ${name}`;
  if (c.includes('curator') || c.includes('allocator')) return `Deposit $${capital} into a ${name} vault on ${chain} and keep the shares`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain}`;
  if (c.includes('rwa')) return `Mint $${capital} of the protocol's yield asset on ${chain} and hold it`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy`;
  if (c.includes('basis')) return `Deposit $${capital} into ${name}'s basis position on ${chain} and keep it open`;
  if (c.includes('farm')) return `Deposit $${capital} into ${name}'s farm on ${chain} and keep the position open`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open`;
}

/**
 * A protocol's farm plan: a dated timeline of six steps, each with a cost
 * and a time. Dates count from the snapshot's as_of day, so the plan is
 * reproducible from the input alone. The verdict line states what the
 * numbers say: farm, thin (a smaller position) or watch only.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null, odds: number, expected: number, asOf: string): string[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const gas = gasTier(chain);
  const action = actionFor(p, capital, chain);
  const verdict = verdictFor(odds, expected);
  const expM = (expected / 1e6).toFixed(expected >= 10_000_000 ? 0 : 1);
  const verdictLine =
    verdict === 'farm'
      ? `Verdict: FARM — odds ${(odds * 100).toFixed(0)}% and expected value $${expM}M clear the farm bar.`
      : verdict === 'thin'
        ? `Verdict: THIN — odds ${(odds * 100).toFixed(0)}% and expected value $${expM}M clear only the thin bar, so size this position small.`
        : `Verdict: WATCH — odds ${(odds * 100).toFixed(0)}% and expected value $${expM}M do not clear the farm bar; track it and revisit if the numbers move.`;
  const d0 = dateAfter(asOf, 0);
  const d1 = dateAfter(asOf, 1);
  const d7 = dateAfter(asOf, 7);
  const dh = dateAfter(asOf, hold);
  const dc = dateAfter(asOf, hold + 3);
  const dr = dateAfter(asOf, hold + 7);
  const steps = [
    `${d0}: Set up a fresh wallet holding only the funds you plan to farm, and keep its seed phrase offline. ${verdictLine} Cost: ${gas.label} for the first transaction. Time: 5 minutes.`,
    `${d0}–${d1}: ${action}. Cost: $${capital} of capital plus ${gas.label}. Time: 15 minutes to set up.`,
    `${d7}: Make a second qualifying action — a deposit, a borrow, a swap or a vote — so the wallet shows repeated use. Cost: ${gas.weekly}. Time: 10 minutes.`,
    `${dh}: Hold checkpoint — keep the position open for at least ${hold} days so the activity lands in more than one snapshot. Cost: the $${capital} stays in the position. Time: 5 minutes to check.`,
    `${dc}: Open the protocol's own points or rewards page and claim only from the link it publishes; never pay anyone to claim. Cost: nothing to check. Time: 10 minutes.`,
    `${dr}: Review the position against this radar's verdict and either keep farming, trim it, or unwind it; log the wallet, dates and every transaction. Cost: ${gas.label} to unwind. Time: 10 minutes.`,
  ];
  return steps.map((s) => {
    let t = s.trim();
    if (t.length > 600) t = `${t.slice(0, 596)}...`;
    if (t.length < 20) t = `${t} Hold the position and check the protocol's own page.`;
    return t;
  });
}

// -------------------------------------------------------------------- radar

/** The live-token protocols, by market cap, for the dashboard's Live token view. */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : 1))
    .slice(0, MODEL.live_rows)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
      multiple: p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0 ? round4(p.mcap_usd / p.tvl_usd) : null,
    }));
}

/** Full accounting: every protocol in the snapshot counted exactly once. */
export function accounting(snapshot: Snapshot, board: BoardRow[]): Accounting {
  const traded = snapshot.protocols.filter((p) => !isPreToken(p));
  const pre = snapshot.protocols.filter((p) => isPreToken(p));
  let cat = 0;
  let ver = 0;
  let wrap = 0;
  let exch = 0;
  for (const p of pre) {
    const r = exclusionReason(p);
    if (r === 'category') cat++;
    else if (r === 'versioned') ver++;
    else if (r === 'wrapped-word') wrap++;
    else if (r === 'exchange-prefix') exch++;
  }
  const watch = new Set(snapshot.watchlist);
  return {
    total: snapshot.protocols.length,
    traded: traded.length,
    traded_with_mcap: traded.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0).length,
    pre_token: pre.length,
    ranked: board.length,
    watchlist: snapshot.watchlist.length,
    early: board.filter((r) => !watch.has(r.slug)).length,
    excluded_category: cat,
    excluded_versioned: ver,
    excluded_wrapped_word: wrap,
    excluded_exchange_prefix: exch,
  };
}

/** Everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const { board, valueFit } = buildBoard(snapshot);
  const oddsValidation = validateOdds(snapshot);
  const valueValidation = validateValue(snapshot);
  const live = liveRows(snapshot);
  const acct = accounting(snapshot, board);
  const { global } = categoryRates(snapshot.protocols);
  const watch = new Set(snapshot.watchlist);
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    growth_beta: round4(valueFit.betaGrowth),
    size_slope: round4(valueFit.sizeSlope),
    mean_log_tvl: round4(valueFit.meanLogTvl),
    global_multiple: round4(valueFit.globalGeo),
    global_traded_rate: round4(global),
    odds_validation: oddsValidation,
    value_validation: valueValidation,
    accounting: acct,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: board.filter((r) => !watch.has(r.slug)).length,
    live_size: live.length,
    verdict_rule: `FARM when launch odds ≥ ${MODEL.verdict_farm_odds} and expected value ≥ $${MODEL.verdict_farm_value / 1e6}M; THIN when odds ≥ ${MODEL.verdict_thin_odds} and expected value ≥ $${MODEL.verdict_thin_value / 1e6}M; otherwise WATCH.`,
    rows: board.map((r) => ({
      rank: r.rank,
      slug: r.slug,
      name: r.name,
      category: r.category,
      chains: r.chains,
      tvl_usd: r.tvl_usd,
      change_7d: r.change_7d,
      listed_at: r.listed_at,
      age_days: r.age_days,
      implied_value_usd: r.implied_value_usd,
      raw_value_usd: r.model.raw_value_usd,
      launch_odds: r.model.launch_odds,
      implied_low_usd: r.model.implied_low_usd,
      implied_high_usd: r.model.implied_high_usd,
      peer_multiple: r.model.peer_multiple,
      size_multiplier: r.model.size_multiplier,
      growth_multiplier: r.model.growth_multiplier,
      seasoning_multiplier: r.model.seasoning_multiplier,
      peers: r.model.peers,
      confidence: r.model.confidence,
      verdict: r.model.verdict,
      watchlist: watch.has(r.slug),
      flags: r.model.flags,
      farm_steps: r.farm_steps,
    })),
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

/** Why a board is not one the gate will take, or null. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value`;
  }
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) {
      const t = s.trim();
      if (t.length < 20 || t.length > 600) return `${r.slug} has a farm step of ${t.length} characters`;
    }
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
