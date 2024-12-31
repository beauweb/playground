from typing import Dict, List, Tuple

def get_order_book(symbol: str) -> Dict[str, List[Tuple[float, float]]]:
    """Get current order book data from exchange
    
    Args:
        symbol: Trading pair symbol (e.g. 'BTC/USDT')
        
    Returns:
        Dict with bids and asks, each containing list of (price, amount) tuples
    """
    # TODO: Implement real exchange API integration
    # For now return mock data
    return {
        'bids': [(50000, 1.5), (49900, 2.0), (49800, 2.5)],  # (price, amount) pairs
        'asks': [(50100, 1.2), (50200, 1.8), (50300, 2.2)]
    }