```shell
============================================================
LOADING DATA
============================================================
Initial data size: 73268 records.


============================================================
CLEANING DATA
------------------------------------------------------------
Select only "Developers by profession" from 5th to 95th percentile based on gross yearly compensation in USD and up to 95th percentile based on "YearsCodePro".
============================================================
Cleaned data size: 25908 records.


============================================================
DATA MANIPULATION
============================================================
Number of countries: 150.

Top 10 countries with the lowest gross yearly compensation:
              min       max
Country                    
India      7093.0  433344.0
Belarus    7104.0  108000.0
Brazil     7128.0  373920.0
Turkey     7140.0  260000.0
Palestine  7176.0  240000.0
Israel     7176.0  382628.0
Pakistan   7176.0  300000.0
Egypt      7176.0  200000.0
Morocco    7200.0  240708.0
Australia  7200.0  389241.0.


Top 10 countries with the highest gross yearly compensation:
                                                        min       max
Country                                                              
France                                               7932.0  435108.0
Italy                                                7428.0  435108.0
United States of America                             8000.0  435000.0
India                                                7093.0  433344.0
Ireland                                             13248.0  429984.0
China                                                9010.0  428856.0
Spain                                                7680.0  422304.0
United Kingdom of Great Britain and Northern Ir...   9048.0  422148.0
Bangladesh                                           7368.0  420000.0
Czech Republic                                       7795.0  415716.0.


Top 5 countries with the highest median years of developer experience:

                 mean
Country              
Cape Verde  26.000000
Botswana    15.000000
Mali        14.500000
Fiji        13.000000
Iceland     12.461538.


Last 5 countries with the lowest median years of developer experience:

                                      mean
Country                                   
Algeria                           2.600000
Democratic Republic of the Congo  2.666667
Malawi                            3.000000
Maldives                          3.000000
Tajikistan                        3.000000.


Mean compensation per year per gender:

Man: $79,728.32.
Man;Non-binary, genderqueer, or gender non-conforming: $90,957.98.
Man;Or, in your own words:: $69,208.25.
Man;Or, in your own words:;Non-binary, genderqueer, or gender non-conforming: $87,250.0.
Man;Or, in your own words:;Woman;Non-binary, genderqueer, or gender non-conforming: $88,568.25.
Man;Woman: $31,149.0.
Man;Woman;Non-binary, genderqueer, or gender non-conforming: $65,440.29.
Non-binary, genderqueer, or gender non-conforming: $90,486.58.
Or, in your own words:: $84,989.89.
Or, in your own words:;Non-binary, genderqueer, or gender non-conforming: $103,568.5.
Or, in your own words:;Woman: $38,487.5.
Or, in your own words:;Woman;Non-binary, genderqueer, or gender non-conforming: $97,667.17.
Prefer not to say: $100,222.35.
Woman: $79,130.5.
Woman;Non-binary, genderqueer, or gender non-conforming: $103,724.44.

============================================================
DESCRIPTIVE STATISTICS
============================================================
Standard deviation - gross yearly compensation: $65,072.0.

Variance - gross yearly compensation: $4,234,300,406.0.

Average gross yearly compensation - World Wide: $80,113.73.

Median gross yearly compensation - World Wide: $63,384.0.

Minimum gross yearly compensation - World Wide: $7,093.0.

Maximum gross yearly compensation - World Wide: $435,108.0.


Average gross yearly compensation - USA: $151,139.47.

Median gross yearly compensation - USA: $140,000.0.

Minimum gross yearly compensation - USA: $8,000.0.

Maximum gross yearly compensation - USA: $435,000.0.


Average gross yearly compensation - Croatia: $43,772.46.

Median gross yearly compensation - Croatia: $34,008.0.

Minimum gross yearly compensation - Croatia: $10,650.0.

Maximum gross yearly compensation - Croatia: $245,280.0.


Average gross yearly compensation - India: $45,609.86.

Median gross yearly compensation - India: $25,794.0.

Minimum gross yearly compensation - India: $7,093.0.

Maximum gross yearly compensation - India: $433,344.0.


============================================================
INFERENCIAL STATISTICS
============================================================
Average yearly compensation World Wide - confidence interval: (99.0%), [79070.69, 81156.78]


============================================================
STATIC ANALYSIS
============================================================
Data set sizes: Croatia (95), Philippines (95), Chile (95).


Correlation between Croatia and Philippines: 0.9268.

Correlation between Croatia and Chile: 0.954.

Correlation between Philippines and Chile: 0.9015.


F-statistic: (4.5019), P-value: (0.0119).

Null hypothesis assuumes no statistical significant differences among the means of the groups, p-value < 0.05 is used to reject null hypothesis. There are statistically significant differences among the means of the groups (Croatia, Philippines, and Chile).
```
