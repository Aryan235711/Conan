# Quality Comparison: C007 (Template) vs C001-C006 (Original)

## Summary
**C007 Quality Grade: B+ (Good, but missing some depth)**
**Original Cases Grade: A (Excellent research-grade quality)**

---

## Detailed Comparison

### 1. EVIDENCE QUALITY

**C001-C006 (Original):**
- Evidence has `evidence_meta` with diagnostic_weight, salience_level, misdirection_risk
- Each piece tagged with bias_tags (e.g., "quiet-anchor", "primary-trap")
- Critical evidence marked explicitly
- Example: C006 has 15 evidence items with full metadata

**C007 (Template):**
- ✅ Has 6 evidence items (correct count)
- ✅ Evidence is specific and concrete
- ❌ NO evidence_meta (missing diagnostic weights)
- ❌ NO bias_tags
- ❌ NO critical markers

**Impact:** C007 evidence is usable but lacks the metadata needed for training models to distinguish diagnostic vs misleading evidence.

---

### 2. INSIGHT DEPTH

**C001-C006 (Original):**
- Insights are domain-specific (e.g., "STAGED_SCENE", "INTENTIONAL_ERASURE")
- Transfer rules are abstract and generalizable
- Insights build on each other (C003 requires C001+C002)

**C007 (Template):**
- ✅ Has clear insight: "Physical Deception - Tape Illusion"
- ✅ Transfer rule is specific and actionable
- ❌ Insight is case-specific, not reusable across cases
- ❌ No insight dependencies (requires_all is empty)

**Impact:** C007 insight is good but doesn't contribute to a knowledge graph like original cases.

---

### 3. SOLUTION STRUCTURE

**C001-C006 (Original):**
- Multiple concept_groups in required_concept_rules
- Detailed forbidden_patterns with multiple concept_groups
- insight_usage_rules showing how previous insights apply
- Bayesian probability structure (C006)

**C007 (Template):**
- ✅ Has required_concept_rules (1 rule)
- ✅ Has forbidden_patterns (1 pattern)
- ✅ Has inference_traps (1 trap)
- ❌ NO insight_usage_rules
- ❌ NO Bayesian structure
- ❌ Simpler concept_groups (fewer variations)

**Impact:** C007 solution is functional but less sophisticated for training reasoning models.

---

### 4. CONTRADICTION QUALITY

**C001-C006 (Original):**
- Contradictions use phrase-level keywords ("chair pulled", "notebook open")
- Multiple contradictions per case (C006 has 3)
- Rich descriptions explaining the tension

**C007 (Template):**
- ✅ Has 1 contradiction
- ✅ Keywords are relevant
- ❌ Keywords are single words, not phrases
- ❌ Only 1 contradiction (originals have 1-3)
- ⚠️ Description is generic ("Surface evidence suggests one thing...")

**Impact:** C007 contradiction is adequate but less nuanced than originals.

---

### 5. DETECTIVE QUESTIONS

**C001-C006 (Original):**
- Questions probe specific evidence items
- Questions guide multi-step reasoning
- Questions challenge assumptions
- Example (C004): "What does partial rigor mortis tell you about time of death?"

**C007 (Template):**
- ✅ Has 3 questions
- ⚠️ Questions are generic ("How could X have committed the crime?")
- ❌ Questions don't reference specific evidence
- ❌ Questions don't guide the reasoning process

**Impact:** C007 questions are placeholder-quality, not pedagogical.

---

### 6. ANALYSIS PROTOCOL

**C001-C006 (Original):**
- Protocols are case-specific and actionable
- Example (C004): "Build a timeline using ONLY physical evidence first"
- Protocols teach reasoning methodology

**C007 (Template):**
- ✅ Has 3 protocol steps
- ⚠️ Steps are generic ("Examine all physical evidence...")
- ❌ Not tailored to the tape trick specifically

**Impact:** C007 protocol is generic advice, not case-specific guidance.

---

### 7. HIDDEN TRUTH & FALSE NARRATIVE

**C001-C006 (Original):**
- Hidden truth is detailed and specific
- False narrative is plausible and compelling
- Clear contrast between the two

**C007 (Template):**
- ✅ Hidden truth explains the tape trick clearly
- ✅ False narrative is plausible (suicide)
- ✅ Good contrast between truth and narrative

**Impact:** C007 EXCELS here - this is template-quality.

---

### 8. SCENARIOS

**C001-C006 (Original):**
- 3-4 scenarios per case
- Scenarios are distinct and testable
- Example (C004): Three different timeline interpretations

**C007 (Template):**
- ✅ Has 3 scenarios
- ✅ Scenarios are distinct
- ✅ Includes false narrative, hidden truth, and third-party option

**Impact:** C007 scenarios are good quality.

---

## OVERALL ASSESSMENT

### What C007 Does WELL:
1. ✅ Core insight (tape trick) is brilliant and well-explained
2. ✅ Hidden truth and false narrative are clear and compelling
3. ✅ Evidence is concrete and specific
4. ✅ Basic structure is correct (6 evidence, 3 scenarios, etc.)

### What C007 is MISSING:
1. ❌ Evidence metadata (diagnostic_weight, bias_tags, critical markers)
2. ❌ Sophisticated solution structure (insight_usage_rules, Bayesian)
3. ❌ Case-specific detective questions and analysis protocol
4. ❌ Multiple contradictions with phrase-level keywords
5. ❌ Insight dependencies (requires_all)

---

## RECOMMENDATION FOR TEMPLATE SYSTEM

### Option A: Accept B+ Quality (Fast)
- Use templates as-is for rapid case generation
- 500-1000 cases in days
- Good enough for initial model training
- Iterate and improve later

### Option B: Enhance Templates (Slower but Better)
- Add evidence_meta generation
- Create case-specific detective questions
- Add multiple contradictions per case
- Generate insight_usage_rules
- Result: A-grade cases, but takes longer

### Option C: Hybrid Approach (RECOMMENDED)
1. Generate 500 cases with templates (B+ quality)
2. Hand-curate 50 "gold standard" cases (A quality)
3. Use gold cases for validation/testing
4. Use template cases for bulk training
5. Model learns from volume + quality examples

---

## VERDICT

**C007 is GOOD ENOUGH for training data** if you need volume.

The core insight (tape trick) is excellent. The missing metadata (evidence_meta, bias_tags) can be added programmatically later if needed.

For 500-1000 training cases, templates are the ONLY realistic path. Hand-authoring A-grade cases would take 1-2 days each = 3-5 years total.

**Recommendation:** Proceed with template system, generate 500 cases, then hand-curate 20-30 gold cases for validation.
