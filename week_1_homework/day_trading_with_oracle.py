'''
Ananya Singh
Class: CS 677
Date: 10/04/2024
Homework Problem #1
Description of Problem: We are analyzing the distribution of returns and a number of trading stratagie with $SPY and $SBUX.
'''
import os

def read_stock_data(ticker: str) -> []:
    """
    Reads a CSV file containing stock data for a given ticker symbol, processes each row to extract 'year', 'weekday', and 'returns' values and returns a list of dictionaries.
    """
    input_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
    ticker_file = os.path.join(input_dir, ticker + '.csv')

    try:
        with open(ticker_file) as f:
            lines = f.read().splitlines()

        columns = lines[0].split(',')
        year = columns.index('Year')
        weekday = columns.index('Weekday')
        returns = columns.index('Return')

        print('opened file for ticker: ', ticker)
        stock_data = []
        for line in lines[1:]:
            row = line.split(',')
            year_value = int(row[year])
            weekday_value = row[weekday]
            returns_value = round(float(row[returns]), 5)

            stock_data.append({
                'year': year_value,
                'weekday': weekday_value,
                'returns': returns_value
            })
        return stock_data
    except Exception as e:
        print(e)
        print('failed to read stock data for ticker: ', ticker)

def calculate_mean(numbers: []) -> float:
    """
    Given a list of numbers, calculates the mean of the numbers.
    """
    return sum(numbers) / len(numbers) if numbers else 0

def calculate_standard_deviation(numbers: []) -> float:
    """
    Given a list of numbers, calculates the standard deviation of the numbers.
    """
    if not numbers:
        return 0
    mean = calculate_mean(numbers)
    squared_differences = [(x - mean) ** 2 for x in numbers]
    variance = sum(squared_differences) / len(numbers)
    return variance ** 0.5

def analyze_stock_data(data) -> dict:
    years = {
        2016: {'Monday': [], 'Tuesday': [], 'Wednesday': [], 'Thursday': [], 'Friday': []},
        2017: {'Monday': [], 'Tuesday': [], 'Wednesday': [], 'Thursday': [], 'Friday': []},
        2018: {'Monday': [], 'Tuesday': [], 'Wednesday': [], 'Thursday': [], 'Friday': []},
        2019: {'Monday': [], 'Tuesday': [], 'Wednesday': [], 'Thursday': [], 'Friday': []},
        2020: {'Monday': [], 'Tuesday': [], 'Wednesday': [], 'Thursday': [], 'Friday': []}
    }

    for row in data:
        year = row['year']
        weekday = row['weekday']
        returns = row['returns']

        if year in years:
            if weekday in years[year]:
                years[year][weekday].append(returns)

    results = {}
    for year in years:
        year_results = {}
        for weekday in years[year]:
            returns = years[year][weekday]
            negative_returns = [r for r in returns if r < 0]
            positive_returns = [r for r in returns if r >= 0]

            stats = {
                'mean_return': round(calculate_mean(returns), 5),
                'std_deviation_return': round(calculate_standard_deviation(returns), 5),
                'count_negative_returns': len(negative_returns),
                'mean_negative_return': round(calculate_mean(negative_returns), 5) if negative_returns else 0,
                'std_deviation_negative_return': round(calculate_standard_deviation(negative_returns),
                                                       5) if negative_returns else 0,
                'count_positive_returns': len(positive_returns),
                'mean_positive_return': round(calculate_mean(positive_returns), 5) if positive_returns else 0,
                'std_deviation_positive_return': round(calculate_standard_deviation(positive_returns),
                                                       5) if positive_returns else 0
            }

            year_results[weekday] = stats

        csv_lines = [
            'Day,mean_return,std_deviation_return,count_negative_returns,mean_negative_return,std_deviation_negative_return,count_positive_returns,mean_positive_return,std_deviation_positive_return']
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            stats = year_results[day]
            csv_lines.append(
                f"{day},{stats['mean_return']},{stats['std_deviation_return']},{stats['count_negative_returns']},"
                f"{stats['mean_negative_return']},{stats['std_deviation_negative_return']},"
                f"{stats['count_positive_returns']},{stats['mean_positive_return']},{stats['std_deviation_positive_return']}"
            )

        results[year] = '\n'.join(csv_lines)
    return results

def save_analysis_to_files(ticker: str, analysis_results: dict) -> None:
    """
    Saves analysis results to CSV files, one for each year.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(current_dir, 'analysis')

    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    for year, results in analysis_results.items():
        filename = f"{ticker}_analysis_{year}.csv"
        filepath = os.path.join(analysis_dir, filename)

        try:
            with open(filepath, 'w') as f:
                f.write(results)
            print(f"Successfully wrote analysis for {year} to {filename}")
        except Exception as e:
            print(f"Error writing file {filename}: {e}")

def compare_daily_returns_across_years(ticker: str, start_year: int = 2016, end_year: int = 2020) -> dict:
    '''
    Reads the five analysis files for all the years so it compares negative and positive returns across years.
    Returns a structured dict with the results.
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(current_dir, 'analysis')

    results = {}

    for year in range(start_year, end_year + 1):
        filename = f"{ticker}_analysis_{year}.csv"
        filepath = os.path.join(analysis_dir, filename)

        try:
            with open(filepath, 'r') as f:
                lines = f.read().strip().split('\n')
            data = []
            total_negative = 0
            total_positive = 0

            for line in lines[1:]:
                values = line.split(',')
                day_data = {
                    'Day': values[0],
                    'mean_negative_return': float(values[4]),
                    'mean_positive_return': float(values[7]),
                    'negative_count': int(values[3]),
                    'positive_count': int(values[6])
                }
                total_negative += int(values[3])
                total_positive += int(values[6])
                data.append(day_data)

            year_results = {
                'total_negative': total_negative,
                'total_positive': total_positive,
                'more_negative_overall': total_negative > total_positive,
                'days': {}
            }

            for day_data in data:
                mean_negative = abs(day_data['mean_negative_return'])
                mean_positive = day_data['mean_positive_return']

                comparison = {
                    'mean_negative_return': mean_negative,
                    'mean_positive_return': mean_positive,
                    'loses_more_than_gains': mean_negative > mean_positive,
                    'difference': round(abs(mean_negative - mean_positive), 5),
                    'negative_count': day_data['negative_count'],
                    'positive_count': day_data['positive_count'],
                    'more_negative_days': day_data['negative_count'] > day_data['positive_count']
                }
                year_results['days'][day_data['Day']] = comparison

            results[year] = year_results

        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    return results

def save_comparison_results(ticker: str, analysis_results: dict) -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(current_dir, 'analysis')

    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    filename = f"{ticker}_comparison_analysis.csv"
    filepath = os.path.join(analysis_dir, filename)

    try:
        with open(filepath, 'w') as f:
            f.write("Year,Day,Down_Day_Loss,Up_Day_Gain,Loses_More_Than_Gains,Difference,"
                    "Negative_Count,Positive_Count,More_Negative_Days,Year_Total_Negative,"
                    "Year_Total_Positive,Year_More_Negative\n")

            for year in sorted(analysis_results.keys()):
                year_data = analysis_results[year]
                for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    if day in year_data['days']:
                        data = year_data['days'][day]
                        f.write(f"{year},{day},{data['mean_negative_return']},"
                                f"{data['mean_positive_return']},{data['loses_more_than_gains']},"
                                f"{data['difference']},{data['negative_count']},{data['positive_count']},"
                                f"{data['more_negative_days']},{year_data['total_negative']},"
                                f"{year_data['total_positive']},{year_data['more_negative_overall']}\n")

        print(f"Successfully wrote comparison analysis to {filename}")
    except Exception as e:
        print(f"Error writing comparison file: {e}")

def analyze_aggregate_table(data) -> str:
    """
    Analyzes returns data and computes aggregate statistics across all years
    """
    days = {
        'Monday': [], 'Tuesday': [], 'Wednesday': [], 'Thursday': [], 'Friday': []
    }

    for row in data:
        weekday = row['weekday']
        returns = row['returns']
        if weekday in days:
            days[weekday].append(returns)

    day_results = {}
    for weekday in days:
        returns = days[weekday]
        negative_returns = [r for r in returns if r < 0]
        positive_returns = [r for r in returns if r >= 0]

        stats = {
            'mean_return': round(calculate_mean(returns), 5),
            'std_deviation_return': round(calculate_standard_deviation(returns), 5),
            'count_negative_returns': len(negative_returns),
            'mean_negative_return': round(calculate_mean(negative_returns), 5) if negative_returns else 0,
            'std_deviation_negative_return': round(calculate_standard_deviation(negative_returns),5) if negative_returns else 0,
            'count_positive_returns': len(positive_returns),
            'mean_positive_return': round(calculate_mean(positive_returns), 5) if positive_returns else 0,
            'std_deviation_positive_return': round(calculate_standard_deviation(positive_returns),5) if positive_returns else 0
        }

        day_results[weekday] = stats

    csv_lines = [
        'Day,mean_return,std_deviation_return,count_negative_returns,mean_negative_return,std_deviation_negative_return,count_positive_returns,mean_positive_return,std_deviation_positive_return'
    ]

    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        stats = day_results[day]
        csv_lines.append(
            f"{day},{stats['mean_return']},{stats['std_deviation_return']},{stats['count_negative_returns']},"
            f"{stats['mean_negative_return']},{stats['std_deviation_negative_return']},"
            f"{stats['count_positive_returns']},{stats['mean_positive_return']},{stats['std_deviation_positive_return']}"
        )

    return '\n'.join(csv_lines)

def save_aggregate_analysis(ticker: str, aggregate_results: dict) -> None:
    """
    Saves aggregate analysis results to a CSV file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(current_dir, 'analysis')

    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    filename = f"{ticker}_analysis_aggregate.csv"
    filepath = os.path.join(analysis_dir, filename)

    try:
        with open(filepath, 'w') as f:
            f.write(aggregate_results)
        print(f"Successfully wrote aggregate analysis to {filename}")
    except Exception as e:
        print(f"Error writing aggregate file: {e}")

def perfect_oracle_trading(ticker: str, initial_investment=100.0) -> float:
    """
    Perfect oracle using previously generated analysis files.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(current_dir, 'analysis')

    portfolio_value = initial_investment

    for year in range(2016, 2021):
        filename = f"{ticker}_analysis_{year}.csv"
        filepath = os.path.join(analysis_dir, filename)

        try:
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()

            for line in lines[1:]:
                values = line.split(',')
                pos_return = float(values[7])
                portfolio_value *= (1 + pos_return) # Investment returns are calculated by multiplying your current value by (1 + return rate)

        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    return round(portfolio_value, 2)

def buy_and_hold_oracle_trading(ticker: str, initial_investment=100.0) -> float:
    """
    Buy-and-hold strategy using analysis files.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(current_dir, 'analysis')

    portfolio_value = initial_investment

    for year in range(2016, 2021):
        filename = f"{ticker}_analysis_{year}.csv"
        filepath = os.path.join(analysis_dir, filename)

        try:
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()

            for line in lines[1:]:
                values = line.split(',')
                pos_return = float(values[7])
                neg_return = float(values[4])
                # Investment returns are calculated by multiplying your current value by (1 + return rate)
                portfolio_value *= (1 + pos_return)
                portfolio_value *= (1 + neg_return)

        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    return round(portfolio_value, 2)

def wrong_oracle_trading(ticker: str, skip_best: int = 0, realize_worst: int = 0, initial_investment=100.0) -> float:
    """
    Oracle that gives wrong advice.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(current_dir, 'analysis')

    portfolio_value = initial_investment
    pos_returns = []
    neg_returns = []

    for year in range(2016, 2021):
        filename = f"{ticker}_analysis_{year}.csv"
        filepath = os.path.join(analysis_dir, filename)

        try:
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()

            for line in lines[1:]:
                values = line.split(',')
                pos_return = float(values[7])
                neg_return = float(values[4])
                pos_returns.append(pos_return)
                neg_returns.append(neg_return)

        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    best_returns = sorted(pos_returns)[-skip_best:] if skip_best > 0 else []
    worst_returns = sorted(neg_returns)[:realize_worst] if realize_worst > 0 else []

    for year in range(2016, 2021):
        filename = f"{ticker}_analysis_{year}.csv"
        filepath = os.path.join(analysis_dir, filename)

        try:
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()

            for line in lines[1:]:
                values = line.split(',')
                pos_return = float(values[7])
                neg_return = float(values[4])

                if pos_return in best_returns:
                    continue
                if neg_return in worst_returns:
                    portfolio_value *= (1 + neg_return)
                else:
                    portfolio_value *= (1 + pos_return)

        except Exception as e:
            print(f"Error reading file {filename}: {e}")
    return round(portfolio_value, 2)

if __name__ == '__main__':
    sbux_data = read_stock_data('SBUX')
    analysis_results = analyze_stock_data(sbux_data)
    save_analysis_to_files('SBUX', analysis_results)

    comparison_results = compare_daily_returns_across_years('SBUX')
    save_comparison_results('SBUX', comparison_results)

    aggregate_results = analyze_aggregate_table(sbux_data)
    save_aggregate_analysis('SBUX', aggregate_results)

    spy_data = read_stock_data('SPY')
    spy_analysis_results = analyze_stock_data(spy_data)
    save_analysis_to_files('SPY', spy_analysis_results)

    spy_comparison_results = compare_daily_returns_across_years('SPY')
    save_comparison_results('SPY', spy_comparison_results)

    spy_aggregate_results = analyze_aggregate_table(spy_data)
    save_aggregate_analysis('SPY', spy_aggregate_results)

    sbux_final = perfect_oracle_trading('SBUX')
    spy_final = perfect_oracle_trading('SPY')

    print(f"\nPerfect Oracle Trading Results:")
    print(f"SBUX: ${sbux_final:}")
    print(f"SPY: ${spy_final:}")

    sbux_hold = buy_and_hold_oracle_trading('SBUX')
    spy_hold = buy_and_hold_oracle_trading('SPY')

    print(f"\nBuy and Hold Results (2016-2020):")
    print(f"SBUX: ${sbux_hold:}")
    print(f"SPY: ${spy_hold:}")

    print("\nPart A - Missing Best 10 Days:")
    sbux_a = wrong_oracle_trading('SBUX', skip_best=10)
    spy_a = wrong_oracle_trading('SPY', skip_best=10)
    print(f"SBUX: ${sbux_a:}")
    print(f"SPY: ${spy_a:,}")

    print("\nPart B - Realizing Worst 10 Days:")
    sbux_b = wrong_oracle_trading('SBUX', realize_worst=10)
    spy_b = wrong_oracle_trading('SPY', realize_worst=10)
    print(f"SBUX: ${sbux_b:}")
    print(f"SPY: ${spy_b:}")

    print("\nPart C - Missing Best 5 and Realizing Worst 5 Days:")
    sbux_c = wrong_oracle_trading('SBUX', skip_best=5, realize_worst=5)
    spy_c = wrong_oracle_trading('SPY', skip_best=5, realize_worst=5)
    print(f"SBUX: ${sbux_c:}")
    print(f"SPY: ${spy_c:}")