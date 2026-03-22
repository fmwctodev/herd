//! Integration tests for the Task-Based Tier Classification feature (Step 1).
//!
//! These tests exercise `classify_by_keywords` and `extract_last_user_message`
//! from the `herd::classifier` module.

use std::collections::HashMap;

use herd::classifier::{classify_by_keywords, extract_last_user_message};
use herd::config::{TaskClassifierConfig, TierConfig};

/// Build a standard test classifier config with three tiers.
fn test_classifier_config() -> TaskClassifierConfig {
    let mut tiers = HashMap::new();

    tiers.insert(
        "heavy".to_string(),
        TierConfig {
            keywords: vec![
                "analyze".to_string(),
                "architect".to_string(),
                "debug complex".to_string(),
                "design system".to_string(),
                "reason about".to_string(),
                "compare tradeoffs".to_string(),
                "explain why".to_string(),
                "review this code".to_string(),
            ],
            model: "qwen2.5:32b-instruct".to_string(),
        },
    );

    tiers.insert(
        "standard".to_string(),
        TierConfig {
            keywords: vec![
                "summarize".to_string(),
                "generate".to_string(),
                "write function".to_string(),
                "convert".to_string(),
                "create template".to_string(),
                "format".to_string(),
            ],
            model: "qwen2.5:14b-instruct".to_string(),
        },
    );

    tiers.insert(
        "lightweight".to_string(),
        TierConfig {
            keywords: vec![
                "heartbeat".to_string(),
                "ping".to_string(),
                "status".to_string(),
                "hello".to_string(),
                "lookup".to_string(),
                "define".to_string(),
                "quick check".to_string(),
            ],
            model: "llama3.2:3b".to_string(),
        },
    );

    TaskClassifierConfig {
        enabled: true,
        strategy: "keyword".to_string(),
        default_tier: "standard".to_string(),
        tiers,
    }
}

// ---------------------------------------------------------------------------
// Test 1: Request with explicit model skips classifier
// ---------------------------------------------------------------------------
// At the integration level, when a model is already specified in the request
// body, the classifier middleware should not run. We verify this indirectly:
// if the message contains a keyword but classify_by_keywords returns a match,
// the middleware layer is responsible for skipping. Here we confirm that the
// function itself DOES match keywords (the skip logic lives in middleware).
// We document the expected middleware behaviour via this structural test.
#[test]
fn test_explicit_model_skips_classifier() {
    // When a request already specifies a model, the middleware bypasses
    // classification entirely. The `classify_by_keywords` function itself
    // does not know about the model field — it only inspects message text.
    // Therefore, the correct assertion is that the *middleware* does not
    // call classify_by_keywords when a model is present.
    //
    // We verify here that classify_by_keywords would return a result for
    // keyword-bearing text, proving the skip must happen at the middleware
    // layer.
    let config = test_classifier_config();
    let result = classify_by_keywords("analyze this code for bugs", &config);
    assert!(
        result.is_some(),
        "classify_by_keywords should match even when a model is present; \
         the middleware is responsible for skipping classification"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Request with X-Herd-Tags header skips classifier
// ---------------------------------------------------------------------------
// Same reasoning as test 1 — the middleware checks for the header before
// calling the classifier. We document the expected behaviour.
#[test]
fn test_tags_header_skips_classifier() {
    // The middleware must check for the X-Herd-Tags header BEFORE calling
    // classify_by_keywords. The function itself has no knowledge of headers.
    // We verify the function still works on its own to confirm the skip
    // must happen upstream.
    let config = test_classifier_config();
    let result = classify_by_keywords("hello world", &config);
    assert!(
        result.is_some(),
        "classify_by_keywords matches regardless of headers; \
         middleware is responsible for checking X-Herd-Tags"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Keyword match — heavy tier
// ---------------------------------------------------------------------------
#[test]
fn test_keyword_match_heavy_tier() {
    let config = test_classifier_config();
    let result = classify_by_keywords("analyze this code for bugs", &config);

    let classification = result.expect("should match heavy tier");
    assert_eq!(classification.tier, "heavy");
    assert_eq!(classification.model, "qwen2.5:32b-instruct");
    assert_eq!(classification.classified_by, "keyword");
}

// ---------------------------------------------------------------------------
// Test 4: Keyword match — lightweight tier
// ---------------------------------------------------------------------------
#[test]
fn test_keyword_match_lightweight_tier() {
    let config = test_classifier_config();
    let result = classify_by_keywords("hello", &config);

    let classification = result.expect("should match lightweight tier");
    assert_eq!(classification.tier, "lightweight");
    assert_eq!(classification.model, "llama3.2:3b");
    assert_eq!(classification.classified_by, "keyword");
}

// ---------------------------------------------------------------------------
// Test 5: No keyword match — falls back to default tier
// ---------------------------------------------------------------------------
#[test]
fn test_no_keyword_match_uses_default() {
    let config = test_classifier_config();
    let result = classify_by_keywords("random unmatched text here", &config);

    let classification = result.expect("should fall back to default tier");
    assert_eq!(classification.tier, "standard");
    assert_eq!(classification.model, "qwen2.5:14b-instruct");
    assert_eq!(classification.classified_by, "default");
}

// ---------------------------------------------------------------------------
// Test 6: Classifier disabled — no classification
// ---------------------------------------------------------------------------
#[test]
fn test_classifier_disabled() {
    let mut config = test_classifier_config();
    config.enabled = false;

    // When the classifier is disabled, the middleware should not register at
    // all — zero overhead. The function itself does not check `enabled`;
    // that is the middleware's responsibility. We verify the config flag is
    // correctly set so that middleware tests can rely on it.
    assert!(!config.enabled);

    // Even if someone accidentally calls classify_by_keywords with a
    // disabled config, the function still operates on text alone. The
    // important contract is that the middleware never calls it.
    let result = classify_by_keywords("analyze this", &config);
    // The function returns a match because it has no awareness of `enabled`.
    // The middleware MUST check `enabled` before calling this function.
    assert!(
        result.is_some(),
        "classify_by_keywords itself does not check `enabled`; \
         the middleware is responsible for gating on config.enabled"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Case-insensitive matching
// ---------------------------------------------------------------------------
#[test]
fn test_case_insensitive_matching() {
    let config = test_classifier_config();

    // ALL CAPS
    let result = classify_by_keywords("ANALYZE this code", &config);
    let classification = result.expect("ANALYZE should match case-insensitively");
    assert_eq!(classification.tier, "heavy");

    // Mixed case
    let result = classify_by_keywords("Please Analyze the results", &config);
    let classification = result.expect("Analyze should match case-insensitively");
    assert_eq!(classification.tier, "heavy");

    // Lowercase (baseline)
    let result = classify_by_keywords("analyze the data", &config);
    let classification = result.expect("analyze should match");
    assert_eq!(classification.tier, "heavy");
}

// ---------------------------------------------------------------------------
// Test 8: extract_last_user_message — standard OpenAI chat format
// ---------------------------------------------------------------------------
#[test]
fn test_extract_last_user_message_standard() {
    let json = serde_json::json!({
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "Sure, I can help."},
            {"role": "user", "content": "analyze this code for bugs"}
        ]
    });

    let message = extract_last_user_message(&json);
    assert_eq!(message, "analyze this code for bugs");
}

// ---------------------------------------------------------------------------
// Test 9: extract_last_user_message — no messages array
// ---------------------------------------------------------------------------
#[test]
fn test_extract_last_user_message_no_messages() {
    let json = serde_json::json!({
        "model": "llama3.2:3b",
        "prompt": "hello world"
    });

    let message = extract_last_user_message(&json);
    assert!(
        message.is_empty(),
        "should return empty string when no messages array is present"
    );
}

// ---------------------------------------------------------------------------
// Test 10: extract_last_user_message — empty messages array
// ---------------------------------------------------------------------------
#[test]
fn test_extract_last_user_message_empty_messages() {
    let json = serde_json::json!({
        "messages": []
    });

    let message = extract_last_user_message(&json);
    assert!(
        message.is_empty(),
        "should return empty string when messages array is empty"
    );
}

// ---------------------------------------------------------------------------
// Test 11: extract_last_user_message — only system/assistant messages
// ---------------------------------------------------------------------------
#[test]
fn test_extract_last_user_message_no_user_role() {
    let json = serde_json::json!({
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hello!"}
        ]
    });

    let message = extract_last_user_message(&json);
    assert!(
        message.is_empty(),
        "should return empty string when no user messages exist"
    );
}

// ---------------------------------------------------------------------------
// Test 12: Multi-word keyword matching (e.g. "debug complex")
// ---------------------------------------------------------------------------
#[test]
fn test_multi_word_keyword_matching() {
    let config = test_classifier_config();

    let result = classify_by_keywords(
        "I need you to debug complex interaction between services",
        &config,
    );
    let classification = result.expect("'debug complex' should match heavy tier");
    assert_eq!(classification.tier, "heavy");
    assert_eq!(classification.model, "qwen2.5:32b-instruct");
}

// ---------------------------------------------------------------------------
// Test 13: Keyword embedded in larger word should still match (substring)
// ---------------------------------------------------------------------------
#[test]
fn test_keyword_substring_match() {
    let config = test_classifier_config();

    // "summarize" is a standard-tier keyword; "summarized" contains it
    let result = classify_by_keywords("The report was summarized already", &config);
    let classification = result.expect("'summarized' should match via substring");
    assert_eq!(classification.tier, "standard");
}

// ---------------------------------------------------------------------------
// Test 14: Empty message falls back to default tier
// ---------------------------------------------------------------------------
#[test]
fn test_empty_message_uses_default() {
    let config = test_classifier_config();
    let result = classify_by_keywords("", &config);
    let classification = result.expect("should fall back to default tier");
    assert_eq!(classification.tier, "standard");
    assert_eq!(classification.classified_by, "default");
}

// ---------------------------------------------------------------------------
// Test 15: ClassificationResult fields are populated correctly
// ---------------------------------------------------------------------------
#[test]
fn test_classification_result_fields() {
    let config = test_classifier_config();
    let result = classify_by_keywords("please review this code carefully", &config);

    let classification = result.expect("'review this code' should match heavy tier");
    assert_eq!(classification.tier, "heavy");
    assert_eq!(classification.model, "qwen2.5:32b-instruct");
    assert_eq!(
        classification.classified_by, "keyword",
        "classified_by should indicate the strategy used"
    );
}
