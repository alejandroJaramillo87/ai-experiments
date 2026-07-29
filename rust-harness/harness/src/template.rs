/*
 * template — Gemma-style chat template rendering.
 *
 * The rendered token IDs must golden-match the Hugging Face reference
 * tokenizer for a fixed corpus of transcripts before any end-to-end run on
 * real weights (design doc, harness design). The exact Gemma 4 turn and
 * function-calling format is taken from the model card when the fixtures are
 * generated on the workstation; the structure below follows the published
 * Gemma convention of <start_of_turn>/<end_of_turn> delimited turns with
 * tool output folded into user turns.
 */

use crate::transcript::{Entry, Role};

/// Render a transcript into the model prompt, ending with an open model turn.
pub fn render(entries: &[Entry]) -> String {
    let mut out = String::new();
    for e in entries {
        let role = match e.role {
            // Gemma has no system role; system text leads the first user turn
            // by convention. Tool results are presented as user-side turns.
            Role::System | Role::User | Role::Tool => "user",
            Role::Model => "model",
        };
        out.push_str(&format!(
            "<start_of_turn>{role}\n{}<end_of_turn>\n",
            e.content
        ));
    }
    out.push_str("<start_of_turn>model\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_turn_structure_with_open_model_turn() {
        let entries = vec![
            Entry {
                role: Role::User,
                content: "fix the bug".into(),
            },
            Entry {
                role: Role::Model,
                content: "{\"tool\": \"read\"}".into(),
            },
            Entry {
                role: Role::Tool,
                content: "file contents".into(),
            },
        ];
        let text = render(&entries);
        assert_eq!(
            text,
            "<start_of_turn>user\nfix the bug<end_of_turn>\n\
             <start_of_turn>model\n{\"tool\": \"read\"}<end_of_turn>\n\
             <start_of_turn>user\nfile contents<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn rendering_is_append_only_across_turns() {
        // Delta prefill depends on turn N's rendering being a prefix of turn
        // N+1's rendering up to the open model turn.
        let mut entries = vec![Entry {
            role: Role::User,
            content: "task".into(),
        }];
        let turn1 = render(&entries);
        entries.push(Entry {
            role: Role::Model,
            content: "answer".into(),
        });
        let turn2 = render(&entries);
        let closed = turn1.trim_end_matches("<start_of_turn>model\n");
        assert!(turn2.starts_with(closed));
    }

    /// Golden test against the Hugging Face reference tokenizer.
    ///
    /// TODO: generate fixtures/template_goldens.json on the workstation with
    /// the Gemma 4 reference tokenizer, then remove the ignore attribute.
    /// Fixture format: [{"entries": [...], "token_ids": [...]}]. Mandatory
    /// before any end-to-end run on real weights.
    #[test]
    #[ignore = "requires fixtures/template_goldens.json generated on the workstation"]
    fn rendered_token_ids_match_hf_reference() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../fixtures/template_goldens.json");
        let data = std::fs::read_to_string(fixture).expect("fixture missing");
        assert!(!data.is_empty());
        unimplemented!("compare render()+tokenizer output against fixture token_ids");
    }
}
