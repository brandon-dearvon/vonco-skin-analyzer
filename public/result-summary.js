(function (root, factory) {
  "use strict";

  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.VonResultSummary = api;
  }
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var DEFAULT_PLAN_INTRO = "Every source-supported match from your visible appearance profile, with visible priorities first.";
  var MAINTENANCE_PLAN_INTRO = "Every source-supported maintenance match from subtle findings in your visible appearance profile.";

  function cleanText(value) {
    return typeof value === "string" ? value.trim().slice(0, 180) : "";
  }

  function uniqueIds(value) {
    return Array.isArray(value) ? value.filter(function (id, index, values) {
      return typeof id === "string" && values.indexOf(id) === index;
    }) : [];
  }

  function sentenceList(values) {
    if (!values.length) {
      return "";
    }
    if (values.length === 1) {
      return values[0];
    }
    return values.slice(0, -1).join(", ") + " and " + values[values.length - 1];
  }

  function build(data) {
    var safeData = data && typeof data === "object" ? data : {};
    var observations = Array.isArray(safeData.observations) ? safeData.observations : [];
    var map = new Map();
    observations.forEach(function (observation) {
      var id = cleanText(observation && observation.id);
      var label = cleanText(observation && observation.label);
      var level = cleanText(observation && observation.level);
      if (id && label && !map.has(id)) {
        map.set(id, { id: id, label: label, level: level });
      }
    });

    function labels(ids) {
      return uniqueIds(ids).map(function (id) {
        var observation = map.get(id);
        return cleanText(observation && observation.label);
      }).filter(Boolean).slice(0, 2);
    }

    var priorityLabels = labels(safeData.priorities);
    var strengthIds = uniqueIds(safeData.strengths);
    var recommendations = safeData.appearanceRecommendations;
    var recommendationItems = recommendations
      ? (Array.isArray(recommendations.services) ? recommendations.services : []).concat(
        Array.isArray(recommendations.products) ? recommendations.products : []
      )
      : [];
    var subtleIds = [];

    recommendationItems.forEach(function (item) {
      uniqueIds(item && item.matchedObservationIds).forEach(function (id) {
        var observation = map.get(id);
        if (observation && observation.level === "subtle" && subtleIds.indexOf(id) === -1) {
          subtleIds.push(id);
        }
      });
    });
    if (!subtleIds.length) {
      strengthIds.concat(observations.map(function (observation) {
        return observation && observation.id;
      })).forEach(function (id) {
        var observation = map.get(id);
        if (observation && observation.level === "subtle" && subtleIds.indexOf(id) === -1) {
          subtleIds.push(id);
        }
      });
    }

    var subtleLabels = labels(subtleIds);
    var notApparentLabels = labels(strengthIds.filter(function (id) {
      var observation = map.get(id);
      return observation && observation.level === "not_observed";
    }));
    var hasRecommendations = recommendationItems.length > 0;
    var hasVisiblePriorityMatch = recommendationItems.some(function (item) {
      return uniqueIds(item && item.matchedObservationIds).some(function (id) {
        var observation = map.get(id);
        return observation && (observation.level === "visible" || observation.level === "prominent");
      });
    });
    var heading = priorityLabels.length
      ? sentenceList(priorityLabels) + (priorityLabels.length === 1 ? " stands out most." : " stand out most.")
      : "No strong visible priority stands out.";
    var copy;
    var strengthsHeading;

    if (subtleLabels.length) {
      copy = sentenceList(subtleLabels) + " appeared subtle in these photos. " + (hasRecommendations
        ? (priorityLabels.length
          ? "Your full photo-based profile and all supported Von & Co matches are below."
          : "Your full photo-based profile and all supported maintenance matches are below.")
        : "Your full photo-based profile is below.");
      strengthsHeading = notApparentLabels.length
        ? "What appears subtle or not apparent"
        : "What appears subtle";
    } else if (notApparentLabels.length) {
      copy = sentenceList(notApparentLabels) + " did not stand out in these photos. Your full photo-based profile is below.";
      strengthsHeading = "What was not apparent";
    } else {
      copy = "Your full photo-based profile shows what was visible, what appeared subtle, and what could not be assessed clearly. " + (hasRecommendations
        ? "All supported Von & Co matches are below."
        : "No automatic catalog match was added for this profile.");
      strengthsHeading = "What appears subtle or not apparent";
    }

    return {
      heading: heading,
      copy: copy,
      strengthsHeading: strengthsHeading,
      primaryInsightValue: priorityLabels[0] || "No single focus",
      secondaryInsightLabel: subtleLabels.length ? "Subtle finding" : "Not apparent",
      secondaryInsightValue: subtleLabels[0] || notApparentLabels[0] || "No single standout",
      subtleLabels: subtleLabels,
      planIntro: hasRecommendations && !hasVisiblePriorityMatch
        ? MAINTENANCE_PLAN_INTRO
        : DEFAULT_PLAN_INTRO
    };
  }

  return {
    build: build
  };
}));
