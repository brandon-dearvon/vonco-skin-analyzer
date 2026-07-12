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
  observations,
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
assert.deepEqual(maintenance.subtleLabels, ["Visible lines", "Visible surface texture"]);
assert.equal(maintenance.heading, "No strong visible priority stands out.");
assert.equal(
  maintenance.copy,
  "Visible lines and Visible surface texture appeared subtle in these photos. Your full photo-based profile and all supported maintenance matches are below.",
);
assert.equal(maintenance.secondaryInsightLabel, "Subtle finding");
assert.equal(maintenance.secondaryInsightValue, "Visible lines");
assert.match(maintenance.planIntro, /^Every source-supported maintenance match/);

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
assert.equal(mixed.heading, "Visible redness stands out most.");
assert.equal(
  mixed.copy,
  "Visible surface texture and Visible lines appeared subtle in these photos. Your full photo-based profile and all supported Von & Co matches are below.",
);
assert.doesNotMatch(mixed.copy, /maintenance/i);
assert.match(mixed.planIntro, /^Every source-supported match/);
assert.doesNotMatch(mixed.planIntro, /maintenance/i);

process.stdout.write("Result summary regression cases passed\n");
