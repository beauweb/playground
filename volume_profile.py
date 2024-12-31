import pandas as pd
import numpy as np

class VolumeProfileAnalyzer:
    def __init__(self, num_bins=24):
        """Initialize volume profile analyzer"""
        self.num_bins = num_bins
        
    def calculate_volume_profile(self, df, window='D'):
        """Calculate volume profile for the given data frame"""
        # Group data by the specified window
        grouped = df.groupby(pd.Grouper(freq=window))
        
        profiles = []
        for _, period_df in grouped:
            if len(period_df) == 0:
                continue
                
            # Calculate price range and bins
            price_range = period_df['high'].max() - period_df['low'].min()
            bin_size = price_range / self.num_bins
            
            # Create price bins
            bins = np.linspace(period_df['low'].min(), period_df['high'].max(), self.num_bins + 1)
            
            # Calculate volume per bin
            volume_profile = np.zeros(self.num_bins)
            for i in range(len(period_df)):
                price = (period_df['high'].iloc[i] + period_df['low'].iloc[i]) / 2
                volume = period_df['volume'].iloc[i]
                bin_idx = int((price - period_df['low'].min()) / bin_size)
                if bin_idx == self.num_bins:  # Edge case for max price
                    bin_idx -= 1
                volume_profile[bin_idx] += volume
            
            # Find POC (Point of Control)
            poc_idx = np.argmax(volume_profile)
            poc_price = bins[poc_idx]
            
            # Calculate Value Area (70% of volume)
            total_volume = np.sum(volume_profile)
            threshold = total_volume * 0.7
            sorted_idx = np.argsort(volume_profile)[::-1]
            value_area_idx = []
            cumulative_volume = 0
            
            for idx in sorted_idx:
                value_area_idx.append(idx)
                cumulative_volume += volume_profile[idx]
                if cumulative_volume >= threshold:
                    break
            
            # Get value area high/low
            value_area_idx = sorted(value_area_idx)
            va_high = bins[max(value_area_idx) + 1]
            va_low = bins[min(value_area_idx)]
            
            profiles.append({
                'timestamp': period_df.index[0],
                'poc': poc_price,
                'va_high': va_high,
                'va_low': va_low,
                'volume_profile': volume_profile.tolist(),
                'price_bins': bins.tolist()
            })
            
        return profiles
    
    def get_support_resistance(self, profiles, num_levels=3):
        """Identify support and resistance levels from volume profiles"""
        # Combine POCs from recent profiles
        all_pocs = [p['poc'] for p in profiles]
        all_vas = [(p['va_high'], p['va_low']) for p in profiles]
        
        # Cluster nearby levels
        levels = []
        for poc in all_pocs:
            found_cluster = False
            for i, level in enumerate(levels):
                if abs(poc - level['price']) / level['price'] < 0.01:  # 1% threshold
                    levels[i]['strength'] += 1
                    found_cluster = True
                    break
            if not found_cluster:
                levels.append({'price': poc, 'strength': 1})
        
        # Sort by strength and return top levels
        levels.sort(key=lambda x: x['strength'], reverse=True)
        return levels[:num_levels]