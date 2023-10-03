
# WeBull API Bot

![GitHub](https://img.shields.io/github/license/M1NL1TE/WeBullAPI_Bot)
![GitHub last commit](https://img.shields.io/github/last-commit/M1NL1TE/WeBullAPI_Bot)
![GitHub issues](https://img.shields.io/github/issues-raw/M1NL1TE/WeBullAPI_Bot)

## Overview

*** THIS PROJECT IS DEPRACATED DUE TO WEBULL CHANGING THEIR API ENDPOINTS AND BLOCKING PROXY SERVER TRAFFIC. THEY ALSO NO LONGER HAVE A WEB BASED TRADING PLATFORM FOR EASY ACCESS TO VIEWING NETWORK TRAFFIC***

This project is a trading bot that interacts with the WeBull API to automate trading tasks and gather financial data. The bot is designed to perform actions such as placing orders, retrieving account information, and analyzing market data.

**Note**: This project is based on the [tedchou12/webull](https://github.com/tedchou12/webull) repository, which provides essential API functionality for interacting with the WeBull platform. We've built upon this foundation to create a customized trading bot.

## Features

- **Automated Trading**: Execute buy and sell orders programmatically based on predefined strategies.
- **Account Information**: Retrieve account details, including balances and positions.
- **Market Data Analysis**: Analyze real-time market data to make informed trading decisions.
- **Customizable Strategies**: Implement and customize trading strategies to fit your investment goals.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Read [HowToFixErrors](HowToFixErrors) files to fix any dependency issues
- Dependencies listed in `requirements.txt`
- A WeBull account with API access (refer to [WeBull API documentation](https://developer.webull.com/api-doc/prepare/api_apply/) for details on obtaining API credentials)

### Installation

1. Clone this repository:

   ```shell
   git clone https://github.com/YourUsername/YourRepository.git
   ```

2. Install the required Python packages:

   ```shell
   pip install -r requirements.txt
   ```

### Usage

1. Obtain your WeBull API credentials, including `client_id` and `access_token`.
2. Run the bot:

   ```shell
   python trading_bot.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the [tedchou12/webull](https://github.com/tedchou12/webull) repository for providing the foundation for the WeBull API integration.

## Contact

Have questions or suggestions? Feel free to [open an issue](https://github.com/M1NL1TE/WeBullAPI_Bot/issues) on GitHub.



