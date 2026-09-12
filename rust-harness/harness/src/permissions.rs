/*
 * permissions — YAML rule file evaluated before each tool dispatch.
 *
 * First matching rule wins; no rule means Ask. A rule matches when its tool
 * field equals the call's tool name (or `*`) and its optional pattern
 * wildcard-matches the call's permission target (a normalized path for file
 * tools, the command for bash).
 *
 * Two scope limits are deliberate and must not be mistaken for enforcement:
 *
 * 1. Bash is not confined by file-tool rules. A deny rule on the `write` tool
 *    for a path outside the workspace does not stop `bash: tee <that path>`;
 *    the bash tool only runs with the workspace as its working directory.
 *    Confining it would need a sandbox, not a rule.
 * 2. A bash pattern is a wildcard match over the whole command line, and `**`
 *    matches shell metacharacters. `bash "cargo **" allow` therefore permits
 *    `cargo test; rm -rf ~`. Any prefix-allow for bash is an allow-anything
 *    rule; use `ask` for bash unless that is the intent.
 */

use anyhow::Result;
use serde::Deserialize;

use crate::tools::ToolCall;
use crate::util::wildcard_match;

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Decision {
    Allow,
    Deny,
    Ask,
}

// Unknown keys are refused rather than dropped: a misspelled `pattern` would
// leave the field None, and the match-everything branch in `evaluate` would
// widen a scoped rule into a blanket one, in the permissive direction for an
// allow rule.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Rule {
    tool: String,
    #[serde(default)]
    pattern: Option<String>,
    action: Decision,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Permissions {
    rules: Vec<Rule>,
}

const KNOWN_TOOLS: [&str; 7] = ["read", "write", "edit", "bash", "glob", "todo", "*"];

impl Permissions {
    /// How many rules the file supplied. Zero means every call asks.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn from_yaml(text: &str) -> Result<Self> {
        let perms: Permissions = serde_yaml::from_str(text)?;
        // A typoed tool name would otherwise be a rule that silently never
        // matches — reject it at load time instead.
        for rule in &perms.rules {
            anyhow::ensure!(
                KNOWN_TOOLS.contains(&rule.tool.as_str()),
                "permission rule names unknown tool: {}",
                rule.tool
            );
        }
        Ok(perms)
    }

    pub fn evaluate(&self, call: &ToolCall) -> Decision {
        for rule in &self.rules {
            let tool_matches = rule.tool == "*" || rule.tool == call.name();
            let target = call.permission_target();
            let pattern_matches = rule
                .pattern
                .as_deref()
                .is_none_or(|p| wildcard_match(p, &target));
            if tool_matches && pattern_matches {
                return rule.action;
            }
        }
        Decision::Ask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULES: &str = "
rules:
  - tool: bash
    pattern: \"rm **\"
    action: deny
  - tool: bash
    pattern: \"cargo **\"
    action: allow
  - tool: bash
    action: ask
  - tool: write
    pattern: \"/etc/**\"
    action: deny
  - tool: \"*\"
    action: allow
";

    fn perms() -> Permissions {
        Permissions::from_yaml(RULES).unwrap()
    }

    #[test]
    fn first_matching_rule_wins() {
        let p = perms();
        let rm = ToolCall::Bash {
            command: "rm -rf build".into(),
        };
        assert_eq!(p.evaluate(&rm), Decision::Deny);
        let cargo = ToolCall::Bash {
            command: "cargo test --workspace".into(),
        };
        assert_eq!(p.evaluate(&cargo), Decision::Allow);
        let other = ToolCall::Bash {
            command: "curl example.com".into(),
        };
        assert_eq!(p.evaluate(&other), Decision::Ask);
    }

    #[test]
    fn patterns_apply_to_file_targets() {
        let p = perms();
        let etc = ToolCall::Write {
            path: "/etc/passwd".into(),
            content: String::new(),
        };
        assert_eq!(p.evaluate(&etc), Decision::Deny);
        let local = ToolCall::Write {
            path: "src/lib.rs".into(),
            content: String::new(),
        };
        assert_eq!(p.evaluate(&local), Decision::Allow);
    }

    #[test]
    fn wildcard_tool_rule_catches_the_rest() {
        let p = perms();
        assert_eq!(
            p.evaluate(&ToolCall::Read { path: "x".into() }),
            Decision::Allow
        );
        assert_eq!(
            p.evaluate(&ToolCall::Todo { items: vec![] }),
            Decision::Allow
        );
    }

    #[test]
    fn unknown_tool_name_in_a_rule_is_rejected_at_load() {
        let err = Permissions::from_yaml("rules:\n  - tool: bahs\n    action: deny\n").unwrap_err();
        assert!(err.to_string().contains("bahs"));
    }

    #[test]
    fn misspelled_rule_key_is_rejected_at_load_naming_the_key() {
        // A dropped `pattern` leaves the field None, and the match-everything
        // branch then turns a scoped allow rule into an allow-anything rule.
        let err = Permissions::from_yaml(
            "rules:\n  - tool: bash\n    patern: \"cargo **\"\n    action: allow\n",
        )
        .unwrap_err();
        assert!(err.to_string().contains("patern"), "{err}");
        let p = Permissions::from_yaml(
            "rules:\n  - tool: bash\n    pattern: \"cargo **\"\n    action: allow\n",
        )
        .unwrap();
        assert_eq!(
            p.evaluate(&ToolCall::Bash {
                command: "cargo test".into()
            }),
            Decision::Allow
        );
        assert_eq!(
            p.evaluate(&ToolCall::Bash {
                command: "rm -rf /home/user".into()
            }),
            Decision::Ask
        );
    }

    #[test]
    fn unknown_top_level_key_in_the_rule_file_is_rejected_at_load() {
        let err = Permissions::from_yaml("rules: []\nrulez:\n  - tool: bash\n").unwrap_err();
        assert!(err.to_string().contains("rulez"), "{err}");
    }

    #[test]
    fn unknown_action_fails_deserialization() {
        assert!(Permissions::from_yaml("rules:\n  - tool: bash\n    action: maybe\n").is_err());
    }

    #[test]
    fn empty_rule_set_defaults_to_ask() {
        let p = Permissions::from_yaml("rules: []").unwrap();
        assert_eq!(
            p.evaluate(&ToolCall::Read { path: "x".into() }),
            Decision::Ask
        );
    }
}
