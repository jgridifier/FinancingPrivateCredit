from financing_private_credit.indicators import get_indicator

indicator = get_indicator("prime_leverage_cycle")
data = indicator.fetch_data("2000-01-01")
result = indicator.calculate(data)


print(result.metadata["current_regime"])