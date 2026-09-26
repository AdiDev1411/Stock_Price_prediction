# Decision Log

## 1. Removed Global News Sentiment from the dashboard
Reason: the dashboard was showing a zero or unhelpful value, and the request was to remove it from `app.py`. Keeping it visible added noise without improving the stock view.

## 2. Reworked RSI calculation to a standard Wilder-style formula
Reason: the earlier RSI logic produced values that stayed too close to the bottom of the chart. A proper RSI implementation is needed so the indicator moves across the expected 0–100 range.

## 3. Locked the RSI chart to a 0–100 scale
Reason: RSI is defined on a fixed 0–100 scale. Free chart scaling makes the indicator harder to interpret and can hide overbought/oversold conditions.

## 4. Added visible RSI bands at 30 and 70
Reason: RSI is usually interpreted using oversold and overbought thresholds. Shaded zones and dashed lines make the chart easier to read.

## 5. Focused the RSI chart on the latest 90 trading days
Reason: the full history made the RSI line appear flat and compressed. Showing a recent window makes short-term momentum easier to see.

## 6. Added the latest RSI annotation on the chart
Reason: the current RSI value should be obvious at a glance without needing to inspect the line position manually.

## 7. Kept the richer stock feature set in the model
Reason: the model benefits from more than just Close, Volume, RSI, and MACD. Additional derived features help the classifier use trend and volatility context.

## 8. Saved model accuracy in metadata
Reason: the dashboard needs a persisted accuracy value so it can display model quality without retraining every time the app starts.

## 9. Added a live accuracy fallback in the app
Reason: if stored metadata is missing or invalid, the dashboard still needs to show a meaningful accuracy estimate instead of zero.

## 10. Used a built-in classifier fallback in training
Reason: the environment did not reliably provide the original ML package set, so using a standard library model keeps training runnable in this workspace.

## 11. Avoided optional sentiment-model dependencies
Reason: the project should still run even if external transformer downloads or heavy sentiment models are unavailable.
