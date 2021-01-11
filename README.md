# WorldQuantAlpha101

使用Python实现的WorldQuant Alpha 101因子，虽然Github上面已经有非常多这种类似的项目，但每个项目对于使用者来说还是较为复杂的，需要单独写本地数据与相应的项目衔接的代码，本项目使用方法非常简单，只需要将输入数据修改为拥有股票代码code，日期date的双重level的MultiLevelIndex即可。

可以通过FactorAnalysis项目中的utils文件读入数据，修改列名排序后输入函数：
import utils
import WorldquantFactors as wf

hist_data = utils.read_data('hist_data20201231.csv')
for c in ['open', 'high', 'low', 'close', 'vwap']:
    hist_data[f'adj_{c}'] = hist_data[c] * hist_data['adj_factor']
hist_data['returns'] = hist_data.sort_index().groupby('code')['adj_close'].pct_change()

material_data = hist_data[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_vwap', 'returns', 'volume', 'industry', 'market_value']]
material_data = material_data.sort_index()
alpha1 = wf.alpha1(material_data)

函数效率较高，计算一个因子从2008-2020年全部时间所有股票的历史数据只需2-3分钟左右。
