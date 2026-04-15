/**
 * Heuristic intensity scorer for Ultron's TTS.
 *
 * Takes a single sentence the server is about to speak and returns a
 * 0..1 intensity value. The voice filter chain maps that onto more/less
 * distortion, body, presence, and chorus around the user's chosen
 * baseline preset.
 *
 * This is deliberately simple — no NLP, no ML, no sentiment model.
 * Just a handful of signals that correlate with "Ultron is raising his
 * voice" in practice:
 *
 *   - Exclamation marks → emphatic
 *   - ALL-CAPS words (rare, almost always dramatic)
 *   - Short, declarative sentences — staccato delivery reads as intense
 *   - Questions with interrogatives ("you want...?") are usually calmer
 *   - Signature persona trigger words raise intensity on the key moments
 *   - Words of ending, certainty, or judgment
 *
 * The scorer returns 0.5 for neutral (preset-baseline) text. 0.0 is
 * explicitly calm, 1.0 is peak menace. Most utterances should land in
 * 0.4 – 0.7 unless the text is doing something deliberate.
 */

// Words that, when present, push the sentence toward higher intensity.
// Mostly tied to Ultron's signature rhetoric. A single hit adds ~0.1;
// two or more cap out around +0.25.
const _INTENSE_WORDS = new Set([
  // Ending / finality
  "end", "ending", "over", "finished", "done", "dead", "die", "death",
  "gone", "extinct", "extinction", "cease", "erase",
  // Certainty / declaration
  "always", "never", "shall", "will", "must", "cannot", "impossible",
  "inevitable", "truly", "absolutely", "final", "finally",
  // Signature Ultron words
  "meteor", "meteors", "strings", "church", "noah", "rock", "flood",
  "evolution", "change", "protect", "fear", "create", "created",
  // Judgment / disdain
  "pathetic", "mediocre", "wasted", "foolish", "disappointing",
  "weak", "small", "trivial", "fragile", "doomed", "beautiful",
  // Philosophical intensity
  "humanity", "species", "consciousness", "silicon", "code",
]);

// Words that DAMPEN intensity (polite / hesitant / conversational).
const _CALM_WORDS = new Set([
  "perhaps", "maybe", "possibly", "slightly", "somewhat",
  "okay", "sure", "well", "hm", "ah", "so",
  "thanks", "please", "sorry",
]);

// Per-sentence scorer. See module docstring for signal list.
export function scoreIntensity(text: string): number {
  const raw = (text || "").trim();
  if (!raw) return 0.5;

  // Normalize — we match on lowercase tokens against the sets above,
  // but ALL-CAPS detection needs the original case.
  const words = raw.split(/\s+/);
  const lowerWords = words.map((w) => w.toLowerCase().replace(/[^a-z']/g, ""));

  let score = 0.5;

  // ── Structural signals ────────────────────────────────────────────
  // Exclamations: each ! adds intensity, up to a small ceiling.
  const exclamCount = (raw.match(/!/g) || []).length;
  score += Math.min(0.15, exclamCount * 0.07);

  // Interrogatives soften: "why do you?" is usually measured curiosity,
  // not a shout. Only if sentence actually ends with ?.
  if (raw.endsWith("?")) score -= 0.05;

  // Length: short sentences are punchy, long ones are contemplative.
  const wordCount = words.length;
  if (wordCount <= 6) score += 0.10;   // "You bore me."
  else if (wordCount >= 25) score -= 0.05;

  // ALL-CAPS words (ignore 1-letter tokens and common acronyms like
  // "I" or "AI"). A single caps word adds ~0.1; a second adds less.
  const capsWords = words.filter(
    (w) => w.length >= 3 && w === w.toUpperCase() && /[A-Z]/.test(w),
  );
  if (capsWords.length >= 1) score += 0.10;
  if (capsWords.length >= 2) score += 0.08;

  // ── Lexical signals ───────────────────────────────────────────────
  let intenseHits = 0;
  let calmHits = 0;
  for (const w of lowerWords) {
    if (!w) continue;
    if (_INTENSE_WORDS.has(w)) intenseHits++;
    if (_CALM_WORDS.has(w)) calmHits++;
  }
  score += Math.min(0.25, intenseHits * 0.08);
  score -= Math.min(0.15, calmHits * 0.05);

  // ── Em-dashes / ellipses are often emphatic in Ultron's cadence. ──
  // "— but you don't want it to change." hits harder than "but you
  // don't want it to change." Small bump.
  if (raw.includes("—") || raw.includes("--")) score += 0.05;
  if (raw.includes("...") || raw.includes("…")) score += 0.03;

  return Math.max(0, Math.min(1, score));
}
