import pandas as pd

df_json = '{"preconditioner":{"8":"none","9":"none","10":"none","11":"none","12":"none","13":"none","14":"none","15":"none"},"executor":{"8":"cuda","9":"cuda","10":"cuda","11":"cuda","12":"cuda","13":"cuda","14":"cuda","15":"cuda"},"nProcs":{"8":8,"9":8,"10":16,"11":16,"12":40,"13":40,"14":80,"15":80},"nCells":{"8":64000000,"9":125000000,"10":64000000,"11":125000000,"12":64000000,"13":125000000,"14":64000000,"15":125000000},"Host":{"8":"hkn","9":"hkn","10":"hkn","11":"hkn","12":"hkn","13":"hkn","14":"hkn","15":"hkn"},"nNodes":{"8":1,"9":1,"10":2,"11":2,"12":5,"13":5,"14":10,"15":10},"TimeStep":{"8":4.7883543078,"9":5.9129097287,"10":8.1222732468,"11":10.69098537,"12":10.9945473544,"13":14.5542214233,"14":15.2778749784,"15":23.303670739},"SolveP":{"8":6.9866913462,"9":8.9118486406,"10":10.7185069875,"11":14.5210142276,"12":12.5392921585,"13":16.8401603357,"14":16.4376090305,"15":25.8779081256},"MomentumPredictor":{"8":0.23510043,"9":0.2164008287,"10":0.5265679139,"11":0.5391900454,"12":1.3252256037,"13":1.3622604259,"14":2.891785388,"15":2.9149989717},"PISOStep":{"8":6.1484186611,"9":7.8346770801,"10":9.7410872094,"11":13.0595248807,"12":12.0032833958,"13":16.0376941323,"14":16.0699228663,"15":24.9921659728},"p_NoIterations":{"8":1.0109062926,"9":0.9926760009,"10":1.0127247432,"11":0.9934161473,"12":1.0113088338,"13":0.9938915396,"14":1.0045795756,"15":0.9868218788},"fvOps":{"8":0.208840018,"9":0.1691214725,"10":0.1231182416,"11":0.0935367476,"12":0.0909541764,"13":0.0687085878,"14":0.065454129,"15":0.0429116945},"fvOpsPIter":{"8":0.2065869206,"9":0.1703692568,"10":0.1215712783,"11":0.0941566612,"12":0.0899370928,"13":0.0691308709,"14":0.0651557434,"15":0.0434847417},"nCellsPerRank":{"8":0.1052631579,"9":0.1052631579,"10":0.2105263158,"11":0.2105263158,"12":0.5263157895,"13":0.5263157895,"14":1.0526315789,"15":1.0526315789},"solver_p":{"8":"GKOCG","9":"GKOCG","10":"GKOCG","11":"GKOCG","12":"GKOCG","13":"GKOCG","14":"GKOCG","15":"GKOCG"}}'
df = pd.DataFrame.read_json(df_json)
    