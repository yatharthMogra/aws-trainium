// build_deck.js -- the KernelTune pitch deck. Every number is from docs/note.md or the README.
//   node build_deck.js   ->   KernelTune.pptx
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/Users/yatharth/.claude/skills/synced/06f7bbd0-6ef6-4237-b779-abf73a19933c_12de7ba0-b6e9-422b-8ad3-48c9c34a3476/pptx/scripts/apply_theme.js");

const THEME = {
  name: "KernelTune",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16212B", lt1: "FFFFFF", dk2: "3D4B57", lt2: "EEF3F2",
    accent1: "0B8F6A", accent2: "D9822B", accent3: "5B6B7A", accent4: "B23A48",
    accent5: "7FC8B2", accent6: "F4D9B5", hlink: "0B8F6A", folHlink: "5B6B7A",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "KernelTune";
pres.author = "Team Skywalker";
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: THEME.colors.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.3, fontSize: 60, bold: true, color: C.background1, align: "left", valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.7, h: 1.2, fontSize: 24, color: C.accent5, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: THEME.colors.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.45, w: 12.1, h: 0.9, fontSize: 36, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "KernelTune  ·  Team Skywalker", options: { x: 0.6, y: 6.95, w: 6, h: 0.3, fontSize: 10, color: C.accent3, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: THEME.colors.accent3, align: "right" },
});

const card = (slide, x, y, w, h, name, fill) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 }, line: { color: fill || C.background2, width: 0 }, objectName: name });
const text = (slide, t, o) => slide.addText(t, Object.assign({ isTextBox: true, margin: 0, color: C.text1, valign: "top" }, o));

// ---------------------------------------------------------------- 1. title
pres.addSection({ title: "Pitch" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Pitch" });
s.addText("KernelTune", { placeholder: "title" });
s.addText("An on-chip kernel autotuner driven by two small models", { placeholder: "body" });
text(s, "Team Skywalker  ·  Hack the Chip, NYU × Annapurna Labs  ·  Kernel optimisation track", { x: 0.8, y: 6.4, w: 11.7, h: 0.4, fontSize: 14, color: C.accent5, objectName: "Byline" });
s.addNotes("A hand-written chip kernel has settings that decide how fast it runs. Pick wrong and it is up to seven times slower, and the right answer changes with every shape and precision. We built KernelTune: a loop where two 8-billion-parameter models tune those settings, with every attempt measured on a real Trainium2 core.");

// ---------------------------------------------------------------- 2. problem
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Pitch" });
s.addText("A hand-written kernel is only fast if its settings are right", { placeholder: "title" });
text(s, [
  { text: "Three settings decide how much data the kernel loads per trip and how much it keeps on the chip.", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "The right values change with every shape and precision. There is no formula.", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "Today an engineer finds them by trial and error.", options: {} },
], { x: 0.6, y: 1.7, w: 5.4, h: 3.6, fontSize: 20, objectName: "Problem text" });
const stats = [
  ["7.1x", "slower at the worst setting than at the best, at an adapter-style shape", C.accent4],
  ["4 of 96", "settings are within 2% of the best at that shape", C.accent2],
  ["0 of 4", "Qwen3-8B layer shapes where AWS's suggested settings run at 512 tokens", C.accent1],
];
stats.forEach(([big, small, col], i) => {
  const y = 1.6 + i * 1.72;
  card(s, 6.5, y, 6.2, 1.5, `Stat card ${i + 1}`);
  text(s, big, { x: 6.8, y: y + 0.2, w: 2.5, h: 1.1, fontSize: 44, bold: true, color: col, valign: "middle", fontFace: THEME.headFontFace, objectName: `Stat ${i + 1}` });
  text(s, small, { x: 9.3, y: y + 0.2, w: 3.2, h: 1.1, fontSize: 15, color: C.text2, valign: "middle", objectName: `Stat label ${i + 1}` });
});
s.addNotes("Before a kernel can compute it has to move data from the chip's large memory into a small on-chip buffer. The three settings say how much to move per trip and how much to keep for reuse. We measured every legal setting: at the 2048 square it barely matters, at an adapter-style shape on Qwen3-8B's layer widths the worst is 7.1 times slower than the best and only 4 of 96 are good. AWS's suggested settings do not run on any of Qwen3-8B's four layer shapes at 512 tokens.");

// ---------------------------------------------------------------- 3. architecture
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Pitch" });
s.addText("Architecture: a loop in which the chip is the judge", { placeholder: "title" });
const boxes = [
  { x: 0.9, y: 1.7, t: "1  Kernel", d: "AWS's blocked matmul, with three block-size settings (M, N, K)", col: C.accent3 },
  { x: 8.1, y: 1.7, t: "2  Trainium2 core", d: "Compile, run, verify against a reference, profile: time, engine busy, bytes moved", col: C.text1 },
  { x: 8.1, y: 4.6, t: "3  Checker", d: "Code builds a menu of untried moves with their measured record. Qwen3-8B picks one", col: C.accent1 },
  { x: 0.9, y: 4.6, t: "4  Optimiser", d: "Qwen3-8B edits the setting. The edit is checked before any chip time is spent", col: C.accent2 },
];
boxes.forEach((b, i) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: b.x, y: b.y, w: 4.3, h: 1.75, rectRadius: 0.1, fill: { color: b.col }, line: { color: b.col, width: 0 }, objectName: `Role box ${i + 1}` });
  text(s, b.t, { x: b.x + 0.25, y: b.y + 0.18, w: 3.8, h: 0.45, fontSize: 20, bold: true, color: C.background1, fontFace: THEME.headFontFace, objectName: `Role title ${i + 1}` });
  text(s, b.d, { x: b.x + 0.25, y: b.y + 0.68, w: 3.8, h: 0.95, fontSize: 14, color: C.background1, objectName: `Role text ${i + 1}` });
});
const arrow = (x, y, w, h, flip, name) => s.addShape(pres.shapes.LINE, Object.assign({ x, y, w, h, line: { color: THEME.colors.dk2, width: 2.5, endArrowType: "triangle" }, objectName: name }, flip));
arrow(5.35, 2.57, 2.6, 0, {}, "Arrow kernel to chip");
arrow(10.25, 3.6, 0, 0.85, {}, "Arrow chip to checker");
arrow(5.35, 5.47, 2.6, 0, { flipH: true }, "Arrow checker to optimiser");
arrow(3.05, 3.6, 0, 0.85, { flipV: true }, "Arrow optimiser to kernel");
text(s, "run it", { x: 5.35, y: 2.15, w: 2.6, h: 0.35, fontSize: 14, color: C.text2, align: "center", objectName: "Label run" });
text(s, "measurements", { x: 10.45, y: 3.82, w: 2.2, h: 0.35, fontSize: 14, color: C.text2, objectName: "Label measurements" });
text(s, "one instruction", { x: 5.35, y: 5.05, w: 2.6, h: 0.35, fontSize: 14, color: C.text2, align: "center", objectName: "Label instruction" });
text(s, "new setting", { x: 1.2, y: 3.82, w: 1.7, h: 0.35, fontSize: 14, color: C.text2, align: "right", objectName: "Label new setting" });
text(s, [
  { text: "10 attempts per run", options: { bold: true, breakLine: true } },
  { text: "Keep the fastest setting found", options: { breakLine: true } },
  { text: "Nothing larger than 8B in the loop", options: {} },
], { x: 5.2, y: 3.35, w: 2.9, h: 1.3, fontSize: 14, color: C.text1, align: "center", valign: "middle", objectName: "Loop rules" });
s.addNotes("Three roles, as our mentor asked for. The workload is AWS's blocked matmul kernel. The checker is code that runs the kernel on the chip and first asks whether the answer is still right, then a small model reads the measurements and says what to change. The optimiser is a second small model that edits the setting. Then we measure again. The models choose three numbers; they do not write kernel code.");

// ---------------------------------------------------------------- 4. what we tried
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Pitch" });
s.addText("What we tried: five checker designs, one model", { placeholder: "title" });
const designs = [
  ["v1", "Raw profile numbers", "2,285 µs", "66 of 100", C.accent4],
  ["v2", "Comparisons worked out in code", "1,553 µs", "70 of 100", C.accent2],
  ["v3", "A menu of measured moves", "1,492 µs", "25 of 100", C.accent3],
  ["v4", "v3, and the edit is checked first", "1,495 µs", "2 of 100", C.accent1],
  ["v5", "v4, with far moves; names its move", "1,494 µs", "0 of 100", C.accent1],
];
designs.forEach(([v, shown, time, wasted, col], i) => {
  const x = 0.6 + i * 2.44;
  card(s, x, 1.6, 2.28, 3.85, `Design card ${v}`);
  s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 1.8, w: 0.7, h: 0.7, fill: { color: col }, line: { color: col, width: 0 }, objectName: `Badge ${v}` });
  text(s, v, { x: x + 0.2, y: 1.8, w: 0.7, h: 0.7, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `Badge text ${v}` });
  text(s, shown, { x: x + 0.2, y: 2.65, w: 1.9, h: 1.0, fontSize: 15, bold: true, objectName: `Shown ${v}` });
  text(s, time, { x: x + 0.2, y: 3.7, w: 1.9, h: 0.5, fontSize: 24, bold: true, color: col, fontFace: THEME.headFontFace, objectName: `Time ${v}` });
  text(s, "median final time", { x: x + 0.2, y: 4.18, w: 1.9, h: 0.3, fontSize: 11, color: C.text2, objectName: `Time label ${v}` });
  text(s, wasted, { x: x + 0.2, y: 4.55, w: 1.9, h: 0.4, fontSize: 18, bold: true, objectName: `Wasted ${v}` });
  text(s, "attempts with no measurement", { x: x + 0.2, y: 4.95, w: 1.9, h: 0.4, fontSize: 11, color: C.text2, objectName: `Wasted label ${v}` });
});
text(s, "Each design fixes a failure we read in the previous one's log. The model, Qwen3-8B, never changed. Adapter-style shape, 10 starting points: best possible 1,361 µs, random guessing 1,467 µs.", { x: 0.6, y: 5.75, w: 12.1, h: 0.8, fontSize: 15, color: C.text2, objectName: "Design caption" });
s.addNotes("Version one showed the model the raw profile. It lost badly to random guessing. The checker said 'increase N to 24, that reduces data movement', the chip came back two and a half times slower, and it said it again. So we changed the message, not the model. Version two showed it comparisons worked out in code. Version three gave it a menu of moves with the measured record for each. Version four caught the second model editing a line it was not asked to touch. Version five had no wasted attempts at all.");

// ---------------------------------------------------------------- 5. what we got
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Pitch" });
s.addText("What we got: the checker decides, not the model", { placeholder: "title" });
s.addImage({ path: path.join(__dirname, "..", "results", "curve_s30_K256_M4096_N12288.png"), x: 0.6, y: 1.55, w: 7.4, h: 5.1, objectName: "Chart: best time against attempts, 30 starts", altText: "Line chart of best kernel time found against attempts on the chip, for each approach, at the adapter shape" });
const got = [
  ["3 → 29 of 30", "runs within 10% of the best, first checker to last, same model", C.accent1],
  ["73%", "of runs within 10% of the best for random guessing", C.accent3],
  ["71% vs 50%", "runs that find the best: a one-line rule on our menu against the 8B model (random 38%)", C.accent2],
];
got.forEach(([big, small, col], i) => {
  const y = 1.55 + i * 1.72;
  card(s, 8.3, y, 4.4, 1.55, `Result card ${i + 1}`);
  text(s, big, { x: 8.55, y: y + 0.15, w: 3.9, h: 0.62, fontSize: 30, bold: true, color: col, fontFace: THEME.headFontFace, valign: "middle", objectName: `Result ${i + 1}` });
  text(s, small, { x: 8.55, y: y + 0.8, w: 3.9, h: 0.65, fontSize: 13, color: C.text2, objectName: `Result label ${i + 1}` });
});
s.addNotes("Same model, same chip, same budget: by redesigning only what the checker says, we went from 3 good runs out of 30 to 29 out of 30. Random guessing gets 73%. Here is the part we did not expect: replace the model with a one-line rule on the same menu and it finds the best setting in 71% of runs. The 8B model manages 50%, random 38%. So the value is in the measured menu. Do not say the model beats random at finding the best: 50% against 38% is not conclusive at 30 runs.");

// ---------------------------------------------------------------- 6. reaches a model
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Pitch" });
s.addText("Tuned, the kernel is level with AWS's compiler", { placeholder: "title" });
const chartOpts = (title) => ({
  barDir: "col", showTitle: true, title, titleFontSize: 14, titleFontFace: "+mn-lt", titleColor: THEME.colors.dk1,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 14, dataLabelFontFace: "+mn-lt", dataLabelColor: THEME.colors.dk1,
  chartColors: [THEME.colors.accent1], showLegend: false,
  catAxisLabelColor: THEME.colors.dk2, valAxisLabelColor: THEME.colors.dk2, catAxisLabelFontSize: 13, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: "DDE3E2", size: 0.75 }, catGridLine: { style: "none" }, valAxisMinVal: 0,
});
s.addChart(pres.charts.BAR, [{ name: "ms", labels: ["Our kernel, untuned", "Tuned by the loop", "AWS compiler"], values: [434.1, 370.9, 372.6] }],
  Object.assign({ x: 0.6, y: 1.55, w: 6.0, h: 4.3, objectName: "Chart: 32-bit layer totals", dataLabelFormatCode: "0.0" }, chartOpts("All 252 projections, 512 tokens, 32-bit (ms)")));
s.addChart(pres.charts.BAR, [{ name: "ms", labels: ["Our kernel, untuned", "Tuned by the loop", "AWS compiler"], values: [8.63, 2.88, 2.75] }],
  Object.assign({ x: 6.75, y: 1.55, w: 6.0, h: 4.3, objectName: "Chart: 16-bit feed-forward block", dataLabelFormatCode: "0.00" }, chartOpts("One feed-forward block, 16-bit, inference time (ms)")));
text(s, "Left: sums of standalone kernel times, one loop run per layer shape. Right: a small model with random weights at Qwen3-8B's sizes; tuned bar is the median of five loop runs (2.86 to 2.89 ms). Level with the compiler at 32-bit, 4% to 5% behind at 16-bit.", { x: 0.6, y: 5.95, w: 12.1, h: 0.85, fontSize: 14, color: C.text2, objectName: "Chart caption" });
s.addNotes("We ran the loop on every linear-layer shape in Qwen3-8B and compared it with AWS's own compiler doing the same multiply with no hand-written kernel. Untuned, a hand-written kernel is 12 to 30 percent slower than the compiler. Tuned by the loop it is level with it on every layer. Then we ran the loop at 16-bit, the precision real models use, scored on real inference time. From as slow as 8.6 milliseconds, all five runs ended at about 2.9, within 5% of the compiler at 2.75. Say 'level with the compiler', never 'faster than'.");

// ---------------------------------------------------------------- 7. limits and next
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Pitch" });
s.addText("What we do not claim, and what comes next", { placeholder: "title" });
card(s, 0.6, 1.6, 6.9, 4.4, "Limits card");
text(s, "Not claimed", { x: 0.9, y: 1.8, w: 6.3, h: 0.5, fontSize: 22, bold: true, color: C.accent4, fontFace: THEME.headFontFace, objectName: "Limits heading" });
text(s, [
  { text: "We do not beat AWS's compiler. Tuned, the kernel ties it at 32-bit and is 4% to 5% behind at 16-bit.", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
  { text: "The 8B model does not beat a one-line rule on the same menu.", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
  { text: "The loop chooses three numbers. It does not write kernel code.", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
  { text: "One operation tested: matmul, which nobody needs to hand-write.", options: { bullet: true } },
], { x: 1.05, y: 2.45, w: 6.2, h: 3.4, fontSize: 18, objectName: "Limits list" });
card(s, 7.8, 1.6, 4.9, 4.4, "Next card", C.text1);
text(s, "Next", { x: 8.1, y: 1.8, w: 4.3, h: 0.5, fontSize: 22, bold: true, color: C.accent5, fontFace: THEME.headFontFace, objectName: "Next heading" });
text(s, "Point KernelTune at the kernels people do write by hand: fused attention, quantised multiplies, custom normalisation.", { x: 8.1, y: 2.4, w: 4.3, h: 1.7, fontSize: 18, color: C.background1, objectName: "Next text" });
text(s, "It needs three things from any kernel: named settings, a reference answer, and the rule for which values are legal.", { x: 8.1, y: 4.25, w: 4.3, h: 1.5, fontSize: 15, color: C.accent5, objectName: "Next requirements" });
s.addNotes("What are we handing over? An on-chip checker this track did not have. Full ground truth at two shapes. And measured evidence of what a small-model loop needs: tell it what the chip measured, not what to think. We do not beat your compiler, and for matmul nobody needs to. The next step is to point KernelTune at the kernels people do write by hand, where there is no compiler version to fall back on.");

(async () => {
  const out = path.join(__dirname, "KernelTune.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
