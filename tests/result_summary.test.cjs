"use strict";

const assert = require("node:assert/strict");
const summaryHelper = require("../public/result-summary.js");

const observations = [
  { id: "visible_lines", label: "Visible lines", level: "subtle" },
  { id: "visible_redness", label: "Visible redness", level: "visible" },
  { id: "pigment_variation", label: "Visible pigment variation", level: "not_observed" },
  { id: "surface_texture", label: "Visible surface texture", level: "subtle" },
  { id: "laxity_appearance", label: "Visible laxity appearance", level: "not_observed" },
];

const maintenancePayload = {
  observations: observations.map((observation) => observation.id === "visible_redness"
    ? { ...observation, level: "not_observed" }
    : observation),
  strengths: ["pigment_variation", "laxity_appearance"],
  priorities: [],
  appearanceRecommendations: {
    services: [
      { matchedObservationIds: ["visible_lines"] },
      { matchedObservationIds: ["surface_texture"] },
      { matchedObservationIds: ["visible_lines"] },
    ],
    products: [
      { matchedObservationIds: ["visible_lines"] },
      { matchedObservationIds: ["surface_texture"] },
      { matchedObservationIds: ["visible_lines", "surface_texture"] },
    ],
  },
};

const maintenance = summaryHelper.build(maintenancePayload);
assert.deepEqual(maintenance.subtleLabels, ["Lines", "Surface texture"]);
assert.equal(maintenance.heading, "Your skin reads balanced overall.");
assert.equal(
  maintenance.copy,
  "Lines and surface texture look soft and understated in these photos. The options below are thoughtful ways to maintain that balance.",
);
assert.equal(maintenance.primaryInsightValue, "Balanced overall");
assert.equal(maintenance.secondaryInsightLabel, "Photo strength");
assert.equal(maintenance.secondaryInsightValue, "Even-looking tone");
assert.equal(maintenance.optionInsightValue, "3 in-studio + 3 skincare");
assert.match(maintenance.planIntro, /^Thoughtful ways to maintain/);

const mixedPayload = {
  observations,
  strengths: ["surface_texture"],
  priorities: ["visible_redness"],
  appearanceRecommendations: {
    services: [{ matchedObservationIds: ["visible_redness"] }],
    products: [{ matchedObservationIds: ["visible_redness"] }],
  },
};

const mixed = summaryHelper.build(mixedPayload);
assert.equal(mixed.heading, "Redness comes into focus.");
assert.equal(
  mixed.copy,
  "Redness is the most noticeable detail in these photos. Smooth-looking texture reads as a photo strength. Your Von & Co options are organized around a calmer, more even-looking tone.",
);
assert.equal(mixed.primaryInsightValue, "Redness");
assert.equal(mixed.secondaryInsightValue, "Smooth-looking texture");
assert.equal(mixed.optionInsightValue, "1 in-studio + 1 skincare");
assert.match(mixed.planIntro, /^Von & Co services and skincare/);
assert.doesNotMatch(mixed.copy + mixed.planIntro, /did not stand out|not apparent|source-supported/i);

const derivedPriority = summaryHelper.build({
  observations,
  strengths: ["pigment_variation"],
  priorities: [],
  appearanceRecommendations: { services: [], products: [] },
});
assert.equal(derivedPriority.heading, "Redness comes into focus.");
assert.equal(derivedPriority.primaryInsightValue, "Redness");
assert.deepEqual(derivedPriority.priorityIds, ["visible_redness"]);

const severityOrdered = summaryHelper.build({
  observations: observations.concat([
    { id: "visible_flaking", label: "Visible flaking", level: "prominent" },
  ]),
  strengths: [],
  priorities: ["visible_redness", "visible_flaking"],
  appearanceRecommendations: { services: [], products: [] },
});
assert.deepEqual(severityOrdered.priorityIds, ["visible_flaking", "visible_redness"]);
assert.equal(severityOrdered.heading, "Flaking and redness come into focus.");
assert.equal(severityOrdered.primaryInsightValue, "Flaking");

process.stdout.write("Result summary regression cases passed\n");
