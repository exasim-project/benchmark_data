import pandas as pd

df_json = '{"nCells":{"3":1000000,"4":1000000,"8":1000000,"12":1000000,"13":1000000,"17":1000000,"18":1000000,"22":1000000,"26":1000000,"30":1000000,"34":1000000,"38":1000000,"41":1000000,"45":1000000,"46":1000000,"166":8000000,"167":8000000,"171":8000000,"175":8000000,"176":8000000,"180":8000000,"181":8000000,"185":8000000,"189":8000000,"193":8000000,"197":8000000,"201":8000000,"204":8000000,"208":8000000,"209":8000000,"329":27000000,"330":27000000,"334":27000000,"338":27000000,"339":27000000,"343":27000000,"344":27000000,"348":27000000,"351":27000000,"355":27000000,"359":27000000,"363":27000000,"366":27000000,"369":27000000,"370":27000000,"490":64000000,"491":64000000,"495":64000000,"499":64000000,"500":64000000,"504":64000000,"505":64000000,"509":64000000,"513":64000000,"517":64000000,"521":64000000,"525":64000000,"528":64000000,"531":64000000,"532":64000000,"649":125000000,"650":125000000,"651":125000000,"652":125000000},"nProcs":{"3":4,"4":4,"8":8,"12":8,"13":8,"17":8,"18":12,"22":16,"26":16,"30":16,"34":32,"38":32,"41":76,"45":76,"46":112,"166":4,"167":4,"171":8,"175":8,"176":8,"180":8,"181":12,"185":16,"189":16,"193":16,"197":32,"201":32,"204":76,"208":76,"209":112,"329":4,"330":4,"334":8,"338":8,"339":8,"343":8,"344":12,"348":16,"351":16,"355":16,"359":32,"363":32,"366":76,"369":76,"370":112,"490":4,"491":4,"495":8,"499":8,"500":8,"504":8,"505":12,"509":16,"513":16,"517":16,"521":32,"525":32,"528":76,"531":76,"532":112,"649":4,"650":8,"651":12,"652":112},"executor":{"3":"cuda","4":"dpcpp","8":"CPU","12":"cuda","13":"dpcpp","17":"hip","18":"dpcpp","22":"CPU","26":"cuda","30":"hip","34":"CPU","38":"hip","41":"CPU","45":"cuda","46":"CPU","166":"cuda","167":"dpcpp","171":"CPU","175":"cuda","176":"dpcpp","180":"hip","181":"dpcpp","185":"CPU","189":"cuda","193":"hip","197":"CPU","201":"hip","204":"CPU","208":"cuda","209":"CPU","329":"cuda","330":"dpcpp","334":"CPU","338":"cuda","339":"dpcpp","343":"hip","344":"dpcpp","348":"CPU","351":"cuda","355":"hip","359":"CPU","363":"hip","366":"CPU","369":"cuda","370":"CPU","490":"cuda","491":"dpcpp","495":"CPU","499":"cuda","500":"dpcpp","504":"hip","505":"dpcpp","509":"CPU","513":"cuda","517":"hip","521":"CPU","525":"hip","528":"CPU","531":"cuda","532":"CPU","649":"dpcpp","650":"dpcpp","651":"dpcpp","652":"CPU"},"preconditioner":{"3":"none","4":"none","8":"none","12":"none","13":"none","17":"none","18":"none","22":"none","26":"none","30":"none","34":"none","38":"none","41":"none","45":"none","46":"none","166":"none","167":"none","171":"none","175":"none","176":"none","180":"none","181":"none","185":"none","189":"none","193":"none","197":"none","201":"none","204":"none","208":"none","209":"none","329":"none","330":"none","334":"none","338":"none","339":"none","343":"none","344":"none","348":"none","351":"none","355":"none","359":"none","363":"none","366":"none","369":"none","370":"none","490":"none","491":"none","495":"none","499":"none","500":"none","504":"none","505":"none","509":"none","513":"none","517":"none","521":"none","525":"none","528":"none","531":"none","532":"none","649":"none","650":"none","651":"none","652":"none"},"TimeStep":{"3":1269.3541,"4":1113.78865,"8":1423.90605,"12":3594.888,"13":907.56155,"17":656.75655,"18":8960.13,"22":772.4503,"26":5049.546,"30":595.60385,"34":635.55915,"38":765.42485,"41":271.54155,"45":26536.44,"46":107.44885,"166":9402.527,"167":11631.505,"171":37805.27,"175":11551.574,"176":5402.725,"180":3600.109,"181":28371.28,"185":17323.0185,"189":16221.8025,"193":3313.7345,"197":15157.871,"201":4995.5015,"204":11815.3985,"208":60449.665,"209":4872.519,"329":37299.32,"330":48272.41,"334":203598.5,"338":31311.635,"339":30883.135,"343":11233.5395,"344":57502.915,"348":126515.72,"351":34505.31,"355":9229.6785,"359":118887.565,"363":10706.1655,"366":100969.955,"369":101330.625,"370":47640.31,"490":91951.8,"491":134850.555,"495":625633.4,"499":70405.975,"500":73861.28,"504":27986.01,"505":117911.6133333333,"509":391501.9,"513":66505.31,"517":23989.6,"521":367910.05,"525":25670.235,"528":334223.3,"531":152342.365,"532":174279.525,"649":320617.8181818182,"650":168610.275,"651":227618.3,"652":444850.5},"SolveP":{"3":404.205,"4":433.898475,"8":633.234025,"12":1668.501,"13":390.097675,"17":246.47415,"18":4433.464,"22":345.463875,"26":2444.04825,"30":256.157475,"34":283.429625,"38":347.095925,"41":112.654875,"45":13242.4965,"46":45.1393,"166":2521.60555,"167":4517.49875,"171":18242.339,"175":4604.9825,"176":2047.130275,"180":1133.993975,"181":13706.6815,"185":8265.18275,"189":7335.8325,"193":1241.922,"197":7200.3305,"201":2116.4425,"204":5577.069,"208":29889.46,"209":2276.31175,"329":9662.82225,"330":19399.193,"334":99571.5825,"338":11245.2185,"339":13084.3755,"343":3383.8205,"344":27043.8825,"348":61838.9025,"351":14382.0185,"355":3191.64225,"359":58085.585,"363":3986.9315,"366":49264.1075,"369":49440.9375,"370":23154.175,"490":25020.822,"491":56480.975,"495":307555.975,"499":23622.4805,"500":31223.5125,"504":8741.07275,"505":54826.1433333333,"509":192403.175,"513":26301.895,"517":8642.5585,"521":180741.295,"525":9609.78125,"528":164235.8775,"531":73279.775,"532":85507.915,"649":139001.5863636364,"650":73125.4416666667,"651":105920.515,"652":219281.2916666667},"MomentumPredictor":{"3":290.7961578947,"4":147.8157894737,"8":104.1670526316,"12":158.6036842105,"13":72.4848947368,"17":109.4654736842,"18":51.2536842105,"22":52.476,"26":93.8783157895,"30":53.1674736842,"34":41.3388947368,"38":43.0687894737,"41":21.0083157895,"45":21.3130526316,"46":8.3990526316,"166":2693.3710526316,"167":1596.7410526316,"171":868.501,"175":1471.4668421053,"176":836.0964210526,"180":876.5373684211,"181":602.2924210526,"185":486.7735263158,"189":940.1478947368,"193":518.3981578947,"197":450.8052631579,"201":460.5241052632,"204":373.7711578947,"208":372.3408947368,"209":189.8405263158,"329":11981.8773684211,"330":5831.7426315789,"334":2964.7978947368,"338":5380.7573684211,"339":2845.21,"343":2969.2489473684,"344":2042.8636842105,"348":1763.6473684211,"351":3404.3894736842,"355":1770.65,"359":1655.3410526316,"363":1672.3436842105,"366":1424.2263157895,"369":1428.33,"370":807.1855263158,"490":28130.9736842105,"491":13342.4052631579,"495":7040.0663157895,"499":14823.4421052632,"500":6830.7963157895,"504":7030.4484210526,"505":4863.1221428571,"509":4168.0626315789,"513":8334.1736842105,"517":4180.3868421053,"521":3929.8094736842,"525":3955.5110526316,"528":3425.4689473684,"531":3427.5642105263,"532":1992.3157894737,"649":26555.63,"650":13733.2909090909,"651":9502.9933333333,"652":3893.8336363636},"PISOStep":{"3":472.852325,"4":473.3073,"8":655.4291,"12":1708.632825,"13":411.6673,"17":269.448075,"18":4450.36175,"22":357.822,"26":2471.07175,"30":268.892625,"34":295.055675,"38":359.036225,"41":122.2272,"45":13254.19725,"46":48.60525,"166":3202.799,"167":4922.649,"171":18430.7545,"175":4959.28175,"176":2241.2155,"180":1324.106775,"181":13853.22125,"185":8388.8965,"189":7580.9465,"193":1369.239875,"197":7324.15475,"201":2238.1685,"204":5693.28125,"208":30010.1675,"209":2327.8505,"329":12126.87475,"330":20870.635,"334":100195.145,"338":12650.73625,"339":13846.6735,"343":4009.19275,"344":27601.7675,"348":62274.9775,"351":15323.97,"355":3627.784,"359":58513.345,"363":4414.70825,"366":49681.1575,"369":49860.9125,"370":23367.435,"490":30738.9675,"491":59973.1975,"495":309012.4,"499":27038.85,"500":33092.065,"504":10195.09325,"505":56229.6833333333,"509":193428.55,"513":28556.0475,"517":9668.0295,"521":181750.9925,"525":10619.10825,"528":165197.9875,"531":74252.895,"532":86027.2975,"649":145757.0227272727,"650":76706.35,"651":108535.73,"652":220290.3333333333},"Host":{"3":"hkn","4":"i20","8":"nla","12":"hkn","13":"i20","17":"nla","18":"i20","22":"nla","26":"hkn","30":"nla","34":"nla","38":"nla","41":"hkn","45":"hkn","46":"i20","166":"hkn","167":"i20","171":"nla","175":"hkn","176":"i20","180":"nla","181":"i20","185":"nla","189":"hkn","193":"nla","197":"nla","201":"nla","204":"hkn","208":"hkn","209":"i20","329":"hkn","330":"i20","334":"nla","338":"hkn","339":"i20","343":"nla","344":"i20","348":"nla","351":"hkn","355":"nla","359":"nla","363":"nla","366":"hkn","369":"hkn","370":"i20","490":"hkn","491":"i20","495":"nla","499":"hkn","500":"i20","504":"nla","505":"i20","509":"nla","513":"hkn","517":"nla","521":"nla","525":"nla","528":"hkn","531":"hkn","532":"i20","649":"i20","650":"i20","651":"i20","652":"i20"},"solver_p":{"3":"GKOCG","4":"GKOCG","8":"PCG","12":"GKOCG","13":"GKOCG","17":"GKOCG","18":"GKOCG","22":"PCG","26":"GKOCG","30":"GKOCG","34":"PCG","38":"GKOCG","41":"PCG","45":"GKOCG","46":"PCG","166":"GKOCG","167":"GKOCG","171":"PCG","175":"GKOCG","176":"GKOCG","180":"GKOCG","181":"GKOCG","185":"PCG","189":"GKOCG","193":"GKOCG","197":"PCG","201":"GKOCG","204":"PCG","208":"GKOCG","209":"PCG","329":"GKOCG","330":"GKOCG","334":"PCG","338":"GKOCG","339":"GKOCG","343":"GKOCG","344":"GKOCG","348":"PCG","351":"GKOCG","355":"GKOCG","359":"PCG","363":"GKOCG","366":"PCG","369":"GKOCG","370":"PCG","490":"GKOCG","491":"GKOCG","495":"PCG","499":"GKOCG","500":"GKOCG","504":"GKOCG","505":"GKOCG","509":"PCG","513":"GKOCG","517":"GKOCG","521":"PCG","525":"GKOCG","528":"PCG","531":"GKOCG","532":"PCG","649":"GKOCG","650":"GKOCG","651":"GKOCG","652":"PCG"}}'
df = pd.DataFrame.read_json(df_json)
    