/*
 * envelope.rs -- the provenance fields every written artifact carries.
 *
 * `common/reports.py` requires a stamp and a git revision on each report the
 * repository writes, and more than one Rust writer now produces one. These
 * live here rather than beside any single writer so the civil-date arithmetic
 * has exactly one implementation to be wrong in.
 */

/// The commit the writing binary was built from, or `"unknown"`.
///
/// Shelling out rather than baking it in at compile time: a stale build stamp
/// naming a commit the binary is not from is worse than an honest absence, and
/// an artifact written outside a checkout should say so rather than claim the
/// last revision anyone compiled at.
pub fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// The current time as `YYYY-MM-DDTHH:MM:SSZ`.
///
/// Rendered without a time library: the workspace's only need for one is this
/// stamp, and a date crate is a dependency, a lockfile entry and a supply
/// chain for eighteen lines of arithmetic.
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let (h, m, s) = (secs % 86_400 / 3600, secs % 3600 / 60, secs % 60);
    // Civil-date conversion (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_now_renders_a_plausible_iso_timestamp() {
        let t = utc_now();
        assert_eq!(t.len(), 20);
        assert!(t.ends_with('Z') && t.contains('T'), "{t}");
        assert!(t.starts_with("20"), "{t}");
    }
}
