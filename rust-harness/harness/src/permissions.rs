/*
 * permissions — YAML rule file evaluated before each tool dispatch.
 *
 * First matching rule wins; no rule means Ask. A rule matches when its tool
 * field equals the call's tool name (or `*`) and its optional pattern
 * wildcard-matches the call's target (path for file tools, command for bash).
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

#[derive(Debug, Deserialize)]
struct Rule {
    tool: String,
    #[serde(default)]
    pattern: Option<String>,
    action: Decision,
}

#[derive(Debug, Deserialize)]
pub struct Permissions {
    rules: Vec<Rule>,
}

impl Permissions {
    pub fn from_yaml(text: &str) -> Result<Self> {
        Ok(serde_yaml::from_str(text)?)
    }

    pub fn evaluate(&self, call: &ToolCall) -> Decision {
        for rule in &self.rules {
            let tool_matches = rule.tool == "*" || rule.tool == call.name();
            let pattern_matches = rule
                .pattern
                .as_deref()
                .is_none_or(|p| wildcard_match(p, call.target()));
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
    fn empty_rule_set_defaults_to_ask() {
        let p = Permissions::from_yaml("rules: []").unwrap();
        assert_eq!(
            p.evaluate(&ToolCall::Read { path: "x".into() }),
            Decision::Ask
        );
    }
}
