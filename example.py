import utils import WorldquantFactors as wf

hist_data = utils.read_data('hist_data20201231.csv') 
for c in ['open', 'high', 'low', 'close', 'vwap']: 
    hist_data[f'adj_{c}'] = hist_data[c] * hist_data['adj_factor'] 
    hist_data['returns'] = hist_data.sort_index().groupby('code')['adj_close'].pct_change()

material_data = hist_data[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_vwap', 'returns', 'volume', 'industry', 'market_value']] 
material_data = material_data.sort_index() 
alpha1 = wf.alpha1(material_data)
