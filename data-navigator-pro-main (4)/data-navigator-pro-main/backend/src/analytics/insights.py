"""
Insights Generator - AI-Powered Natural Language Insights
Converts analyzer results into human-readable insights using Gemini

When Gemini is unavailable, generates REAL data-driven insights by
actually reading and interpreting the numbers in the analysis results.
"""

import google.generativeai as genai
import json
import os
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime


class InsightsGenerator:
    """
    Generates natural language insights from analytics results.
    - Primary: Gemini AI (rich, contextual insights)
    - Fallback: Rule-based engine that reads actual data values
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.model = genai.GenerativeModel(model_name)
            self.ai_enabled = True
            print(f"✅ Insights Generator initialized (Gemini AI enabled with {model_name})")
        else:
            self.ai_enabled = False
            self.model = None
            print("⚠️  GOOGLE_API_KEY not found — using data-driven rule-based insights.")

    async def generate_insights(
        self,
        analysis_result: Dict[str, Any],
        insight_type: Literal["detailed", "executive"] = "detailed",
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        analysis_type = analysis_result.get("analysis_type", "unknown")

        if self.ai_enabled:
            try:
                return await self._generate_with_ai(analysis_result, insight_type)
            except Exception as e:
                print(f"Gemini failed ({e}), using rule-based fallback.")

        # Always use real-data fallback
        return self._generate_data_insights(analysis_result, insight_type)

    # ─────────────────────────────────────────────
    # GEMINI AI PATH
    # ─────────────────────────────────────────────

    async def _generate_with_ai(self, result: Dict[str, Any], insight_type: str) -> Dict[str, Any]:
        analysis_type = result.get("analysis_type", "unknown")

        # Summarise the result so the prompt stays lean
        summary = self._summarise_for_prompt(result)

        mode = "brief executive summary (3 bullet points)" if insight_type == "executive" \
               else "detailed analyst report (5–6 bullet points)"

        prompt = f"""You are a senior business data analyst. 
Analyze the following {analysis_type} results and write a {mode}.

DATA:
{json.dumps(summary, indent=2, default=str)}

Rules:
- Every insight MUST reference specific numbers, percentages, or dates from the data.
- Focus on business impact: revenue, growth, risk, opportunity, what to act on.
- No generic phrases like "the data shows interesting patterns" — be concrete.
- Return ONLY a JSON array of strings, no markdown, no extra text.

Example format:
["Revenue grew 18.3% to $2.4M in the last 30 days, driven by strong weekend performance.",
 "Product A accounts for 42% of total sales — over-reliance risk if it declines.",
 "March is historically 23% below average — plan promotions in advance."]
"""
        response = self.model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        insights = json.loads(text)

        return {
            "analysis_type": analysis_type,
            "insight_type": insight_type,
            "generated_at": datetime.utcnow().isoformat(),
            "insights": insights,
            "metadata": {"ai_powered": True}
        }

    def _summarise_for_prompt(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Trim large result objects to keep prompts focused."""
        analysis_type = result.get("analysis_type")
        out = {"analysis_type": analysis_type}

        if analysis_type == "kpi":
            out["as_of_date"] = result.get("as_of_date")
            out["comparison_period"] = result.get("comparison_period")
            out["kpis"] = result.get("kpis", [])

        elif analysis_type == "time_series":
            out["granularity"] = result.get("granularity")
            out["date_range"] = result.get("date_range")
            out["summary"] = result.get("summary", {})
            # Include last 6 data points only
            data = result.get("data", [])
            out["recent_data"] = data[-6:] if len(data) > 6 else data

        elif analysis_type == "dimension_breakdown":
            out["dimension"] = result.get("dimension")
            out["metrics"] = result.get("metrics")
            out["top_5"] = result.get("data", [])[:5]
            out["summary"] = result.get("summary", {})

        elif analysis_type == "forecast":
            out["metric"] = result.get("metric")
            out["method"] = result.get("method")
            out["forecast_horizon"] = result.get("forecast_horizon")
            out["accuracy_metrics"] = result.get("accuracy_metrics")
            fc = result.get("forecast", [])
            out["first_forecast"] = fc[0] if fc else None
            out["last_forecast"] = fc[-1] if fc else None
            out["mid_forecast"] = fc[len(fc)//2] if fc else None

        elif analysis_type == "anomaly_detection":
            out["metric"] = result.get("metric")
            out["total_anomalies"] = result.get("total_anomalies")
            out["anomaly_rate"] = result.get("anomaly_rate")
            out["baseline_statistics"] = result.get("baseline_statistics")
            out["top_anomalies"] = result.get("anomalies", [])[:5]

        elif analysis_type == "seasonality":
            out["metric"] = result.get("metric")
            out["overall_average"] = result.get("overall_average")
            dow = result.get("patterns", {}).get("day_of_week", [])
            mon = result.get("patterns", {}).get("month", [])
            out["best_day"] = max(dow, key=lambda x: x["avg_value"], default=None) if dow else None
            out["worst_day"] = min(dow, key=lambda x: x["avg_value"], default=None) if dow else None
            out["best_month"] = max(mon, key=lambda x: x["avg_value"], default=None) if mon else None
            out["worst_month"] = min(mon, key=lambda x: x["avg_value"], default=None) if mon else None

        else:
            out.update(result)

        return out

    # ─────────────────────────────────────────────
    # RULE-BASED DATA-DRIVEN FALLBACK
    # ─────────────────────────────────────────────

    def _generate_data_insights(self, result: Dict[str, Any], insight_type: str) -> Dict[str, Any]:
        analysis_type = result.get("analysis_type", "unknown")

        dispatch = {
            "kpi":                  self._insights_kpi,
            "time_series":          self._insights_time_series,
            "dimension_breakdown":  self._insights_breakdown,
            "forecast":             self._insights_forecast,
            "anomaly_detection":    self._insights_anomalies,
            "seasonality":          self._insights_seasonality,
        }

        fn = dispatch.get(analysis_type, self._insights_generic)
        insights = fn(result, insight_type)

        return {
            "analysis_type": analysis_type,
            "insight_type": insight_type,
            "generated_at": datetime.utcnow().isoformat(),
            "insights": insights,
            "metadata": {"ai_powered": False, "engine": "rule_based_data_driven"}
        }

    # ── KPI ──────────────────────────────────────

    def _insights_kpi(self, result: Dict[str, Any], mode: str) -> List[str]:
        kpis = result.get("kpis", [])
        if not kpis:
            return ["No KPI data available — ensure the dataset has numeric metric columns."]

        insights = []
        positive_moves, negative_moves = [], []

        for k in kpis:
            name   = k.get("metric", "metric").replace("_", " ").title()
            cur    = k.get("current_value")
            prev   = k.get("previous_value")
            chg    = k.get("change_percent")
            unit   = k.get("unit") or ""
            dirn   = k.get("direction", "positive")

            if cur is None:
                continue

            cur_fmt  = self._fmt(cur, unit)
            prev_fmt = self._fmt(prev, unit) if prev is not None else None

            if chg is not None:
                arrow = "▲" if chg > 0 else "▼"
                vs    = f" vs previous period ({prev_fmt})" if prev_fmt else ""
                good  = (chg > 0 and dirn == "positive") or (chg < 0 and dirn == "negative")
                flag  = "✅" if good else "⚠️"
                insights.append(
                    f"{flag} {name} is {cur_fmt} — {arrow} {abs(chg):.1f}%{vs}."
                )
                if good:
                    positive_moves.append((name, chg))
                else:
                    negative_moves.append((name, abs(chg)))
            else:
                insights.append(f"📊 {name}: {cur_fmt} (no prior period for comparison).")

        if positive_moves:
            best = max(positive_moves, key=lambda x: abs(x[1]))
            insights.append(f"🚀 Strongest mover: {best[0]} with a {abs(best[1]):.1f}% improvement.")

        if negative_moves:
            worst = max(negative_moves, key=lambda x: abs(x[1]))
            insights.append(f"🔴 Needs attention: {worst[0]} declined {worst[1]:.1f}% — investigate root cause and consider corrective action.")

        if not negative_moves and len(kpis) > 1:
            insights.append("✅ All tracked metrics are moving in the right direction this period.")

        return insights[:6] if mode == "detailed" else insights[:3]

    # ── TIME SERIES ───────────────────────────────

    def _insights_time_series(self, result: Dict[str, Any], mode: str) -> List[str]:
        summary = result.get("summary", {})
        date_range = result.get("date_range", {})
        granularity = result.get("granularity", "day")
        data = result.get("data", [])

        if not summary:
            return ["No time-series summary available."]

        insights = []
        dr = f"{str(date_range.get('start',''))[:10]} → {str(date_range.get('end',''))[:10]}"
        insights.append(f"📅 Analysis covers {len(data)} {granularity}s from {dr}.")

        for metric, s in summary.items():
            name  = metric.replace("_", " ").title()
            unit  = s.get("unit") or ""
            avg   = s.get("avg")
            total = s.get("total")
            mn    = s.get("min")
            mx    = s.get("max")
            trend = s.get("trend", "flat")

            arrow = "📈" if trend == "up" else "📉" if trend == "down" else "➡️"
            if total is not None:
                insights.append(
                    f"{arrow} {name}: total {self._fmt(total, unit)}, avg {self._fmt(avg, unit)} per {granularity}. "
                    f"Range: {self._fmt(mn, unit)} – {self._fmt(mx, unit)}. Trend: {trend.upper()}."
                )
            else:
                insights.append(
                    f"{arrow} {name}: avg {self._fmt(avg, unit)} per {granularity}. "
                    f"Range: {self._fmt(mn, unit)} – {self._fmt(mx, unit)}. Trend: {trend.upper()}."
                )

            if trend == "up" and avg and mn:
                pct = ((avg - mn) / mn * 100) if mn else 0
                if pct > 20:
                    insights.append(f"   └ {name} has grown {pct:.0f}% from its low — strong upward momentum.")
            elif trend == "down" and avg and mx:
                pct = ((mx - avg) / mx * 100) if mx else 0
                if pct > 20:
                    insights.append(f"   └ {name} has dropped {pct:.0f}% from its peak — monitor closely.")

        # Highlight most recent vs average
        if data and summary:
            time_col = result.get("time_column", "date")
            last = data[-1]
            for metric, s in list(summary.items())[:2]:
                if metric in last and s.get("avg"):
                    last_val = last[metric]
                    avg_val  = s["avg"]
                    diff_pct = ((last_val - avg_val) / avg_val * 100) if avg_val else 0
                    unit = s.get("unit") or ""
                    name = metric.replace("_", " ").title()
                    if abs(diff_pct) > 5:
                        direction = "above" if diff_pct > 0 else "below"
                        insights.append(
                            f"🔎 Most recent {granularity}: {name} = {self._fmt(last_val, unit)}, "
                            f"which is {abs(diff_pct):.1f}% {direction} the period average."
                        )

        return insights[:7] if mode == "detailed" else insights[:3]

    # ── BREAKDOWN ─────────────────────────────────

    def _insights_breakdown(self, result: Dict[str, Any], mode: str) -> List[str]:
        data      = result.get("data", [])
        dimension = result.get("dimension", "segment")
        metrics   = result.get("metrics", [])
        summary   = result.get("summary", {})

        if not data or not metrics:
            return ["No breakdown data available."]

        metric = metrics[0]
        dim_name = dimension.replace("_", " ").title()
        met_name = metric.replace("_", " ").title()
        met_summary = summary.get("metrics", {}).get(metric, {})
        unit = met_summary.get("unit") or ""
        total = met_summary.get("total")
        insights = []

        top = data[0]
        top_val = top.get(metric, 0)
        top_label = top.get(dimension, "—")

        if total and total > 0:
            share = (top_val / total * 100)
            insights.append(
                f"🥇 Top {dim_name}: **{top_label}** contributes {self._fmt(top_val, unit)} "
                f"({share:.1f}% of total {self._fmt(total, unit)})."
            )
        else:
            insights.append(f"🥇 Top {dim_name}: **{top_label}** with {self._fmt(top_val, unit)}.")

        if len(data) >= 3:
            top3_val = sum(d.get(metric, 0) for d in data[:3])
            if total and total > 0:
                top3_share = top3_val / total * 100
                insights.append(
                    f"📊 Top 3 {dim_name}s account for {top3_share:.1f}% of total {met_name} "
                    f"({self._fmt(top3_val, unit)})."
                )

        if len(data) >= 2:
            bottom = data[-1]
            bot_val = bottom.get(metric, 0)
            bot_label = bottom.get(dimension, "—")
            if top_val and bot_val:
                ratio = top_val / bot_val if bot_val else 0
                insights.append(
                    f"📉 Bottom {dim_name}: **{bot_label}** at {self._fmt(bot_val, unit)} — "
                    f"{ratio:.1f}× less than the top performer."
                )

        if len(data) >= 5 and total:
            top5_val  = sum(d.get(metric, 0) for d in data[:5])
            rest_val  = total - top5_val
            rest_pct  = rest_val / total * 100 if total else 0
            if rest_pct > 30:
                rest_count = met_summary.get("total") or len(data)
                insights.append(
                    f"💡 The remaining {dim_name}s collectively contribute {rest_pct:.1f}% — "
                    f"there may be growth opportunity in the long tail."
                )

        insights.append(
            f"🎯 Focus: Invest in {top_label} to protect revenue leadership and investigate "
            f"what makes it outperform to replicate in other {dim_name.lower()}s."
        )

        return insights[:6] if mode == "detailed" else insights[:3]

    # ── FORECAST ──────────────────────────────────

    def _insights_forecast(self, result: Dict[str, Any], mode: str) -> List[str]:
        metric   = result.get("metric", "metric").replace("_", " ").title()
        fc       = result.get("forecast", [])
        method   = result.get("method", "linear")
        acc      = result.get("accuracy_metrics", {})
        horizon  = result.get("forecast_horizon", len(fc))
        conf     = result.get("confidence_level", 0.95)

        if not fc:
            return ["No forecast data returned — check dataset has enough historical data."]

        first = fc[0]
        last  = fc[-1]
        mid   = fc[len(fc) // 2] if len(fc) > 2 else fc[0]

        first_val = first.get("predicted_value", 0)
        last_val  = last.get("predicted_value", 0)
        pct_chg   = ((last_val - first_val) / first_val * 100) if first_val else 0

        method_label = "Prophet ML" if "prophet" in method else "Linear Trend"
        arrow = "📈" if pct_chg > 0 else "📉"
        direction = "increase" if pct_chg > 0 else "decrease"
        insights = []

        insights.append(
            f"{arrow} {metric} is forecast to {direction} by {abs(pct_chg):.1f}% "
            f"over the next {horizon} days — from {self._fmt(first_val)} to {self._fmt(last_val)}."
        )

        mid_val = mid.get("predicted_value", 0)
        insights.append(
            f"📅 At the midpoint ({mid.get('date','')}), expected value is {self._fmt(mid_val)}."
        )

        if last.get("lower_bound") and last.get("upper_bound"):
            insights.append(
                f"📐 {int(conf*100)}% confidence range at end of period: "
                f"{self._fmt(last['lower_bound'])} – {self._fmt(last['upper_bound'])}."
            )

        if acc.get("mape") is not None:
            mape = acc["mape"]
            quality = "high" if mape < 10 else "moderate" if mape < 20 else "low"
            insights.append(
                f"🎯 Forecast accuracy ({method_label}): MAPE = {mape:.1f}% — {quality} confidence. "
                + ("Plan confidently." if quality == "high" else
                   "Use alongside business judgment." if quality == "moderate" else
                   "Treat as directional only — gather more data for better accuracy.")
            )

        if pct_chg > 0:
            insights.append(
                f"✅ Positive outlook for {metric}. Consider increasing inventory, staffing, "
                f"or marketing spend to capitalise on projected growth."
            )
        else:
            insights.append(
                f"⚠️ Declining {metric} forecast. Identify drivers behind the trend and "
                f"activate retention or revenue-recovery initiatives before the period ends."
            )

        return insights[:5] if mode == "detailed" else insights[:3]

    # ── ANOMALIES ─────────────────────────────────

    def _insights_anomalies(self, result: Dict[str, Any], mode: str) -> List[str]:
        metric     = result.get("metric", "metric").replace("_", " ").title()
        anomalies  = result.get("anomalies", [])
        total      = result.get("total_anomalies", 0)
        rate       = result.get("anomaly_rate", 0)
        baseline   = result.get("baseline_statistics", {})
        total_pts  = result.get("total_data_points", 0)
        insights   = []

        mean  = baseline.get("mean")
        exp_lo, exp_hi = (baseline.get("expected_range") or [None, None])

        insights.append(
            f"🔍 Scanned {total_pts} data points for {metric}. "
            f"Baseline mean: {self._fmt(mean)}. "
            f"Normal range: {self._fmt(exp_lo)} – {self._fmt(exp_hi)}."
        )

        if total == 0:
            insights.append(f"✅ No anomalies detected — {metric} is behaving normally throughout the period.")
            return insights

        critical = [a for a in anomalies if a.get("severity") == "critical"]
        high     = [a for a in anomalies if a.get("severity") == "high"]

        insights.append(
            f"⚠️ {total} anomalies detected ({rate:.1f}% of data): "
            f"{len(critical)} critical, {len(high)} high, {total - len(critical) - len(high)} other."
        )

        for a in anomalies[:3]:
            date   = a.get("date", "unknown date")
            val    = a.get("actual_value")
            dev    = a.get("deviation_percent", 0)
            dirn   = a.get("direction", "above")
            sev    = a.get("severity", "").upper()
            icon   = "🔴" if sev == "CRITICAL" else "🟠" if sev == "HIGH" else "🟡"
            insights.append(
                f"{icon} {date}: {metric} = {self._fmt(val)} — {abs(dev):.1f}% {dirn} normal. [{sev}]"
            )

        # Clustering insight
        if total >= 3:
            dates = [a.get("date","") for a in anomalies[:5]]
            insights.append(
                f"🗓️ Anomaly cluster around: {', '.join(d[:10] for d in dates[:3])}. "
                f"Check for external events (promotions, outages, seasonal factors) on these dates."
            )

        above = sum(1 for a in anomalies if a.get("direction") == "above")
        below = total - above
        if above > below:
            insights.append(f"📈 Most anomalies ({above}/{total}) are ABOVE normal — possible demand spikes or data errors worth investigating.")
        elif below > above:
            insights.append(f"📉 Most anomalies ({below}/{total}) are BELOW normal — possible data gaps, system outages, or demand drops.")

        return insights[:6] if mode == "detailed" else insights[:3]

    # ── SEASONALITY ───────────────────────────────

    def _insights_seasonality(self, result: Dict[str, Any], mode: str) -> List[str]:
        metric   = result.get("metric", "metric").replace("_", " ").title()
        overall  = result.get("overall_average", 0)
        patterns = result.get("patterns", {})
        dow      = patterns.get("day_of_week", [])
        mon      = patterns.get("month", [])
        insights = []

        insights.append(f"📊 {metric} overall average: {self._fmt(overall)} per period.")

        if dow:
            best_day  = max(dow, key=lambda x: x["avg_value"])
            worst_day = min(dow, key=lambda x: x["avg_value"])
            insights.append(
                f"📅 Best day: {best_day['day']} (+{best_day['vs_overall_percent']:.1f}% vs average, "
                f"avg {self._fmt(best_day['avg_value'])})."
            )
            insights.append(
                f"📉 Weakest day: {worst_day['day']} ({worst_day['vs_overall_percent']:.1f}% vs average, "
                f"avg {self._fmt(worst_day['avg_value'])})."
            )
            spread = best_day["avg_value"] - worst_day["avg_value"]
            spread_pct = (spread / overall * 100) if overall else 0
            if spread_pct > 30:
                insights.append(
                    f"⚡ Large day-of-week variation ({spread_pct:.0f}% spread). "
                    f"Consider scheduling promotions on {worst_day['day']} to lift the low point."
                )

        if mon:
            best_mon  = max(mon, key=lambda x: x["avg_value"])
            worst_mon = min(mon, key=lambda x: x["avg_value"])
            insights.append(
                f"📆 Strongest month: {best_mon['month']} (+{best_mon['vs_overall_percent']:.1f}%, "
                f"avg {self._fmt(best_mon['avg_value'])})."
            )
            insights.append(
                f"🥶 Weakest month: {worst_mon['month']} ({worst_mon['vs_overall_percent']:.1f}%, "
                f"avg {self._fmt(worst_mon['avg_value'])})."
            )
            insights.append(
                f"💡 Plan inventory, staffing, and campaigns around {best_mon['month']} peak. "
                f"Use {worst_mon['month']} for training, maintenance, or aggressive promotions to offset the dip."
            )

        return insights[:7] if mode == "detailed" else insights[:3]

    # ── GENERIC FALLBACK ──────────────────────────

    def _insights_generic(self, result: Dict[str, Any], mode: str) -> List[str]:
        analysis_type = result.get("analysis_type", "unknown")
        keys = [k for k in result.keys() if k not in ("analysis_type", "method", "schema_auto_detected")]
        return [
            f"Analysis type: {analysis_type.replace('_',' ').title()} completed.",
            f"Data fields available: {', '.join(str(k) for k in keys[:8])}.",
            "Connect Gemini AI (set GOOGLE_API_KEY) for detailed natural-language insights."
        ]

    # ─────────────────────────────────────────────
    # FORMATTING HELPERS
    # ─────────────────────────────────────────────

    def _fmt(self, value, unit: str = "") -> str:
        """Format a number with smart abbreviation."""
        if value is None:
            return "—"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)

        prefix = unit if unit and unit in ("$", "€", "£", "¥") else ""
        suffix = unit if unit and unit not in ("$", "€", "£", "¥") else ""

        if abs(v) >= 1_000_000:
            return f"{prefix}{v/1_000_000:.2f}M{suffix}"
        if abs(v) >= 1_000:
            return f"{prefix}{v/1_000:.1f}K{suffix}"
        if abs(v) < 1 and v != 0:
            return f"{prefix}{v:.3f}{suffix}"
        return f"{prefix}{v:,.2f}{suffix}"