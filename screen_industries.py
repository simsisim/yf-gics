import yfinance as yf
import pandas as pd
import csv
import time

KEEP_COLS = [
    'symbol', 'shortName', 'exchange',
    'regularMarketPrice', 'regularMarketChangePercent',
    'marketCap', 'trailingPE', 'priceToBook',
    'dividendYield', 'epsTrailingTwelveMonths',
    'fiftyTwoWeekChangePercent',
    'fiftyDayAverage', 'twoHundredDayAverage',
    'averageDailyVolume3Month', 'averageAnalystRating',
    'sharesOutstanding',
]

def screener_name(industry_name: str) -> str:
    # Yahoo Finance sector API uses " - " but screener requires em dash "—"
    return industry_name.replace(' - ', '—')


def screen_industry(industry_name, region='us', page_size=250):
    q = yf.EquityQuery('and', [
        yf.EquityQuery('eq', ['industry', industry_name]),
        yf.EquityQuery('eq', ['region', region]),
    ])
    rows = []
    offset = 0
    total = None
    while True:
        result = yf.screen(q, sortField='intradaymarketcap', sortAsc=False,
                           size=page_size, offset=offset)
        quotes = result.get('quotes', [])
        if total is None:
            total = result.get('total', 0)
        rows.extend(quotes)
        offset += len(quotes)
        if offset >= total or not quotes:
            break
        time.sleep(0.3)
    return rows, total


def main():
    industries = pd.read_csv('industries.csv')
    all_rows = []

    for _, row in industries.iterrows():
        sector_key = row['sector_key']
        sector_name = row['sector_name']
        industry_key = row['industry_key']
        industry_name = row['industry_name']

        print(f"  {sector_name} / {industry_name} ... ", end='', flush=True)
        try:
            quotes, total = screen_industry(screener_name(industry_name))
            print(f"{total} stocks")
            for q in quotes:
                record = {col: q.get(col) for col in KEEP_COLS}
                record['sector_key'] = sector_key
                record['sector_name'] = sector_name
                record['industry_key'] = industry_key
                record['industry_name'] = industry_name
                all_rows.append(record)
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.5)

    out_cols = ['symbol', 'shortName', 'sector_name', 'industry_name',
                'sector_key', 'industry_key', 'exchange',
                'regularMarketPrice', 'regularMarketChangePercent',
                'marketCap', 'trailingPE', 'priceToBook',
                'dividendYield', 'epsTrailingTwelveMonths',
                'fiftyTwoWeekChangePercent',
                'fiftyDayAverage', 'twoHundredDayAverage',
                'averageDailyVolume3Month', 'averageAnalystRating',
                'sharesOutstanding']

    df = pd.DataFrame(all_rows, columns=out_cols)
    df.to_csv('stocks_by_industry.csv', index=False)
    print(f"\nDone. {len(df)} stocks saved to stocks_by_industry.csv")


if __name__ == '__main__':
    main()
