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

  var DEFAULT_PLAN_INTRO = "Von & Co services and skincare matched to your photo profile, with the clearest matches first.";
  var MAINTENANCE_PLAN_INTRO = "Thoughtful ways to maintain what already looks balanced and refine the subtle details.";
  var POSITIVE_STRENGTH_LABELS = {
    visible_lines: "Soft-looking lines",
    visible_redness: "Calm-looking tone",
    pigment_variation: "Even-looking tone",
    surface_texture: "Smooth-looking texture",
    pore_visibility: "Refined-looking pores",
    laxity_appearance: "Firm-looking skin",
    blemish_like_spots: "Clear-looking skin",
    scar_like_texture: "Smooth-looking surface",
    superficial_vessels: "Even-looking tone",
    visible_flaking: "Smooth-looking surface"
  };
  var FOCUS_GOALS = {
    visible_lines: "softer-looking lines",
    visible_redness: "a calmer, more even-looking tone",
    pigment_variation: "a brighter, more even-looking tone",
    surface_texture: "smoother-looking texture",
    pore_visibility: "refined-looking pores",
    laxity_appearance: "firmer-looking skin",
    blemish_like_spots: "clearer-looking skin",
    scar_like_texture: "smoother-looking texture",
    superficial_vessels: "a clearer, more even-looking tone",
    visible_flaking: "a smoother, more hydrated-looking surface"
  };

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

  function consumerLabel(value) {
    var text = cleanText(value).replace(/^Visible\s+/i, "");
    return text ? text.charAt(0).toUpperCase() + text.slice(1) : "";
  }

  function lowerFirst(value) {
    var text = cleanText(value);
    return text ? text.charAt(0).toLowerCase() + text.slice(1) : "";
  }

  function uniqueText(values) {
    return values.filter(function (value, index, list) {
      return value && list.indexOf(value) === index;
    });
  }

  function naturalList(values) {
    return sentenceList(values.map(function (value, index) {
      return index === 0 ? value : lowerFirst(value);
    }));
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
        return consumerLabel(observation && observation.label);
      }).filter(Boolean).slice(0, 2);
    }

    var statedPriorities = uniqueIds(safeData.priorities);
    var priorityLevelRank = { prominent: 0, visible: 1 };
    var priorityIds = observations.filter(function (observation) {
      return observation && (observation.level === "visible" || observation.level === "prominent");
    }).sort(function (left, right) {
      var levelDifference = priorityLevelRank[left.level] - priorityLevelRank[right.level];
      if (levelDifference) {
        return levelDifference;
      }
      var leftStatedIndex = statedPriorities.indexOf(left.id);
      var rightStatedIndex = statedPriorities.indexOf(right.id);
      var leftRank = leftStatedIndex === -1 ? Number.MAX_SAFE_INTEGER : leftStatedIndex;
      var rightRank = rightStatedIndex === -1 ? Number.MAX_SAFE_INTEGER : rightStatedIndex;
      return leftRank - rightRank;
    }).map(function (observation) {
      return observation.id;
    });
    var priorityLabels = labels(priorityIds);
    var focusGoals = uniqueText(priorityIds.slice(0, 2).map(function (id) {
      return FOCUS_GOALS[id] || "";
    }));
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
    var clearStrengthIds = strengthIds.filter(function (id) {
      var observation = map.get(id);
      return observation && observation.level === "not_observed";
    });
    var positiveStrengths = uniqueText(clearStrengthIds.concat(strengthIds).map(function (id) {
      return POSITIVE_STRENGTH_LABELS[id] || "";
    })).slice(0, 2);
    var hasRecommendations = recommendationItems.length > 0;
    var hasVisiblePriorityMatch = priorityLabels.length > 0 || recommendationItems.some(function (item) {
      return uniqueIds(item && item.matchedObservationIds).some(function (id) {
        var observation = map.get(id);
        return observation && (observation.level === "visible" || observation.level === "prominent");
      });
    });
    var serviceCount = recommendations && Array.isArray(recommendations.services)
      ? recommendations.services.length
      : 0;
    var productCount = recommendations && Array.isArray(recommendations.products)
      ? recommendations.products.length
      : 0;
    var optionParts = [];
    if (serviceCount) {
      optionParts.push(serviceCount + " in-studio");
    }
    if (productCount) {
      optionParts.push(productCount + " skincare");
    }
    var optionInsightValue = optionParts.length ? optionParts.join(" + ") : "Explore in person";
    var heading = priorityLabels.length
      ? naturalList(priorityLabels) + (priorityLabels.length === 1
        ? " comes into focus."
        : " come into focus.")
      : "Your skin reads balanced overall.";
    var copy;

    if (priorityLabels.length) {
      copy = naturalList(priorityLabels) + (priorityLabels.length === 1
        ? " is the most noticeable detail in these photos. "
        : " are the most noticeable details in these photos. ");
      if (positiveStrengths.length) {
        copy += naturalList(positiveStrengths) + (positiveStrengths.length === 1
          ? " reads as a photo strength. "
          : " read as photo strengths. ");
      } else {
        copy += "The rest of your profile looks softer and more understated. ";
      }
      copy += hasRecommendations
        ? (focusGoals.length
          ? "Your Von & Co options are organized around " + naturalList(focusGoals) + "."
          : "Your Von & Co options are organized around what your photos show most clearly.")
        : "Your complete photo profile is below.";
    } else if (subtleLabels.length) {
      copy = naturalList(subtleLabels) + (subtleLabels.length === 1
        ? " looks soft and understated in these photos. "
        : " look soft and understated in these photos. ");
      copy += hasRecommendations
        ? "The options below are thoughtful ways to maintain that balance."
        : "Your complete photo profile is below.";
    } else if (positiveStrengths.length) {
      copy = naturalList(positiveStrengths) + (positiveStrengths.length === 1
        ? " reads as a strength in these photos. "
        : " read as strengths in these photos. ") + "Your complete photo profile is below.";
    } else {
      copy = "The reviewed details look soft and balanced in these photos. Your complete photo profile is below.";
    }

    return {
      heading: heading,
      copy: copy,
      strengthsHeading: "What already looks strong",
      primaryInsightValue: priorityLabels[0] || "Balanced overall",
      secondaryInsightLabel: "Photo strength",
      secondaryInsightValue: positiveStrengths[0] || (subtleLabels[0] ? "Soft-looking " + lowerFirst(subtleLabels[0]) : "Balanced overall"),
      optionInsightValue: optionInsightValue,
      priorityIds: priorityIds,
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
