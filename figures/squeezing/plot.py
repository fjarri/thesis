import numpy
import json
import pickle

from figures import get_path
import figures.mplhelpers as mplh

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.patches as patches
import matplotlib.path as path

from scipy.interpolate import interp1d


def buildRiedelTomographyPath():
    """Process outlined blue line from Fig.3 in Riedel 2010"""
    xstart = 137.35189
    ystart = 90.287253

    xn90 = 137.35189
    x360 = 993.039
    y15 = 109.64235
    y0 = 398.97217

    yscale = (y15 - y0) / 15
    xscale = (x360 - xn90) / (360 + 90)

    """
    points = (
        "0,0 14.50878,-1.568829 24.01256,-1.804477 9.50378,-0.235648 34.72699,2.276605 "
        "64.63501,37.401344 29.90803,35.12473 57.92996,97.02575 73.67738,186.27925 "
        "15.74742,89.2535 26.89408,328.94804 26.89408,328.94804 0,0 11.84222,-239.32902 "
        "27.17539,-324.70817 15.33317,-85.37915 37.30822,-145.80606 66.7308,-182.33072 "
        "29.42259,-36.524652 53.98301,-45.175188 75.48187,-45.648008 21.49886,-0.47282 "
        "42.77044,3.737291 73.61912,38.420108 30.84868,34.68281 51.95238,95.04074 "
        "69.15619,181.47674 17.20381,86.436 28.81508,331.37343 28.81508,331.37343 0,0 "
        "11.86912,-254.44324 30.73609,-338.09694 18.86697,-83.6537 33.03585,-128.95471 "
        "60.85126,-163.90638 27.81541,-34.95167 47.80063,-48.936916 79.15907,-49.548283 "
        "31.35845,-0.611367 50.95538,13.159523 79.60519,47.287693 28.64981,34.12817 "
        "53.63199,120.14375 61.81177,163.28546 8.17978,43.14171 12.82612,84.24291 "
        "12.82612,84.24291")
    """

    # interpolated
    points = ['0,0', '0.0567,-0.006', '0.16526,-0.0177', '0.10859,-0.0115',
       '0.26908,-0.0285', '0.47673,-0.0502', '0.20764,-0.0217',
       '0.46243,-0.0481', '0.7596,-0.0785', '0.29717,-0.0304',
       '0.63673,-0.0648', '1.0139,-0.10248', '0.37718,-0.0377',
       '0.79197,-0.0787', '1.23962,-0.12225', '0.44764,-0.0436',
       '0.92814,-0.0897', '1.43673,-0.13775', '0.5086,-0.048',
       '1.04528,-0.098', '1.60529,-0.14903', '0.56,-0.0511',
       '1.14334,-0.10332', '1.74523,-0.15603', '0.6019,-0.0527',
       '1.22236,-0.10588', '1.85661,-0.1588', '0.63426,-0.0529',
       '1.28231,-0.10559', '1.9394,-0.1573', '0.65709,-0.0517',
       '1.32321,-0.10248', '1.9936,-0.15157', '0.67039,-0.0491',
       '1.34505,-0.0965', '2.01922,-0.14158', '0.67417,-0.0451',
       '1.34784,-0.0877', '2.01625,-0.12733', '0.66842,-0.0396',
       '1.33157,-0.0761', '1.98471,-0.10884', '0.65313,-0.0327',
       '1.29624,-0.0617', '1.92456,-0.0861', '0.62833,-0.0244',
       '1.24186,-0.0444', '1.83585,-0.0591', '0.59399,-0.0147',
       '1.24938,-0.0187', '1.96348,-0.005', '0.7141,0.014',
       '1.48691,0.0461', '2.31573,0.10346', '0.82883,0.0574',
       '1.71367,0.14006', '2.65184,0.25536', '0.93816,0.11529',
       '1.92965,0.2632', '2.97176,0.451', '1.04212,0.1878',
       '2.13486,0.4155', '3.27554,0.69039', '1.14067,0.2749',
       '2.32928,0.59698', '3.56313,0.97354', '1.23385,0.37656',
       '2.51294,0.8076', '3.83458,1.30042', '1.32164,0.49281',
       '2.68582,1.0474', '4.08986,1.67105', '1.40403,0.62365',
       '2.84792,1.31636', '4.32896,2.08543', '1.48105,0.76907',
       '2.99925,1.61449', '4.55192,2.54356', '1.55266,0.92907',
       '3.13979,1.94178', '4.75869,3.04543', '1.6189,1.10365',
       '3.26958,2.29824', '4.94932,3.59105', '1.67975,1.29282',
       '3.38857,2.68386', '5.12378,4.18042', '1.7352,1.49656',
       '3.49679,3.09864', '5.28206,4.81353', '1.78528,1.71489',
       '3.59424,3.5426', '5.4242,5.4904', '1.82996,1.9478',
       '3.68091,4.0157', '5.55016,6.211', '1.86925,2.1953',
       '3.73113,4.49519', '5.58311,6.89981', '1.85198,2.40463',
       '3.69406,4.91399', '5.5237,7.52823', '1.82963,2.61424',
       '3.64683,5.33335', '5.44905,8.15748', '1.80222,2.82414',
       '3.58946,5.75328', '5.3592,8.78759', '1.76973,3.0343',
       '3.52195,6.17377', '5.25412,9.41853', '1.73217,3.24476',
       '3.44429,6.59482', '5.13382,10.05032', '1.68954,3.4555',
       '3.35649,7.01643', '4.99832,10.68295', '1.64183,3.66652',
       '3.25854,7.43861', '4.84759,11.31643', '1.58906,3.87782',
       '3.15045,7.86135', '4.68165,11.95075', '1.5312,4.0894',
       '3.03221,8.28466', '4.50049,12.58592', '1.46827,4.30126',
       '2.90382,8.70852', '4.3041,13.22192', '1.40028,4.51341',
       '2.7653,9.13295', '4.09251,13.85878', '1.32721,4.72583',
       '2.61662,9.55794', '3.8657,14.49648', '1.24907,4.93854',
       '2.4578,9.9835', '3.62366,15.13502', '1.16586,5.15152',
       '2.28884,10.40961', '3.36641,15.77441', '1.07758,5.36479',
       '2.10974,10.83629', '3.09395,16.41463', '0.98421,5.57834',
       '1.95046,11.74435', '2.89713,18.40276', '0.94667,6.65842',
       '1.87377,13.80925', '2.77971,21.35724', '0.90593,7.548',
       '1.7907,15.49315', '2.6527,23.74023', '0.862,8.24707',
       '1.70123,16.79606', '2.5161,25.55172', '0.81487,8.75566',
       '1.60538,17.71799', '2.36992,26.79174', '0.76455,9.07374',
       '1.50313,18.25891', '2.21415,27.46026', '0.71102,9.20134',
       '1.39448,18.41885', '2.04878,27.55729', '0.6543,9.13844',
       '1.27945,18.1978', '1.87383,27.08284', '0.59439,8.88504',
       '1.15802,17.59576', '1.68929,26.03691', '0.53128,8.44114',
       '1.0302,16.61272', '1.49517,24.41947', '0.46496,7.80676',
       '0.89598,15.24869', '1.29144,22.23056', '0.39547,6.98187',
       '0.75538,13.50367', '1.07814,19.47016', '0.32276,5.96649',
       '0.60838,11.37766', '0.85524,16.13827', '0.24687,4.76061',
       '0.45499,8.87066', '0.62277,12.23489', '0.16777,3.36424',
       '0.2952,5.98266', '0.38068,7.76003', '0.0855,1.77736',
       '0.12903,2.71367', '0.12903,2.71367', '0,0', '0.0463,-0.93488',
       '0.13674,-2.70862', '0.0905,-1.77374', '0.22517,-4.38635',
       '0.40205,-7.7418', '0.17688,-3.35546', '0.39594,-7.45376',
       '0.65514,-12.1989', '0.25921,-4.74513', '0.55855,-10.1371',
       '0.896,-16.07989', '0.33744,-5.94278', '0.713,-12.43639',
       '1.12461,-19.38479', '0.41161,-6.94841', '0.85929,-14.35162',
       '1.341,-22.11361', '0.4817,-7.762', '0.99743,-15.88278',
       '1.54515,-24.26633', '0.54771,-8.38356', '1.12742,-17.02988',
       '1.73707,-25.84296', '0.60965,-8.81308', '1.24924,-17.79292',
       '1.91675,-26.8435', '0.66751,-9.05058', '1.36292,-18.1719',
       '2.08421,-27.26794', '0.72128,-9.09605', '1.46844,-18.16682',
       '2.23942,-27.1163', '0.77099,-8.94947', '1.56581,-17.77766',
       '2.38241,-26.38855', '0.81661,-8.61088', '1.65501,-17.00446',
       '2.51317,-25.08472', '0.85815,-8.08026', '1.73606,-15.8472',
       '2.63168,-23.2048', '0.89562,-7.3576', '1.80896,-14.30587',
       '2.73797,-20.74878', '0.92901,-6.44292', '1.8737,-12.38048',
       '2.83202,-17.71668', '0.95832,-5.3362', '1.94259,-10.57492',
       '2.953,-15.71644', '1.01041,-5.14151', '2.04696,-10.18581',
       '3.10985,-15.13316', '1.06289,-4.94734', '2.15212,-9.79773',
       '3.26788,-14.55141', '1.11576,-4.75369', '2.25806,-9.41067',
       '3.42709,-13.97121', '1.16903,-4.56054', '2.36479,-9.02464',
       '3.58748,-13.39255', '1.22269,-4.36791', '2.4723,-8.63963',
       '3.74904,-12.81542', '1.27675,-4.17579', '2.58061,-8.25564',
       '3.9118,-12.23983', '1.33119,-3.98418', '2.6897,-7.87269',
       '4.07573,-11.66577', '1.38603,-3.79309', '2.79958,-7.49076',
       '4.24084,-11.09327', '1.44126,-3.6025', '2.91024,-7.10985',
       '4.40713,-10.52229', '1.49689,-3.41243', '3.02169,-6.72996',
       '4.5746,-9.95285', '1.55291,-3.22288', '3.13393,-6.35111',
       '4.74325,-9.38495', '1.60932,-3.03384', '3.24695,-5.97328',
       '4.91308,-8.81859', '1.66613,-2.8453', '3.36076,-5.59647',
       '5.08409,-8.25376', '1.72333,-2.65729', '3.47536,-5.2207',
       '5.25629,-7.69049', '1.78092,-2.46978', '3.59074,-4.84594',
       '5.42965,-7.12873', '1.83891,-2.28279', '3.65883,-4.4567',
       '5.4602,-6.52653', '1.80136,-2.06984', '3.58417,-4.0356',
       '5.34887,-5.90209', '1.7647,-1.86649', '3.51129,-3.63372',
       '5.2402,-5.30649', '1.72891,-1.67277', '3.44014,-3.25108',
       '5.13415,-4.73974', '1.694,-1.48867', '3.37077,-2.88768',
       '5.03074,-4.20186', '1.65998,-1.31417', '3.30315,-2.54351',
       '4.92998,-3.69281', '1.62682,-1.14931', '3.23729,-2.21858',
       '4.83184,-3.21263', '1.59455,-0.99405', '3.17319,-1.91288',
       '4.73635,-2.7613', '1.56316,-0.84841', '3.11085,-1.62641',
       '4.6435,-2.33881', '1.53264,-0.71239', '3.05026,-1.35918',
       '4.55327,-1.94518', '1.50302,-0.58599', '2.99143,-1.11119',
       '4.4657,-1.5804', '1.47426,-0.46921', '2.93436,-0.88243',
       '4.38075,-1.24448', '1.44638,-0.36204', '2.87905,-0.67291',
       '4.29844,-0.9374', '1.41939,-0.2645', '2.82551,-0.48262',
       '4.21878,-0.65918', '1.39327,-0.17657', '2.77371,-0.31157',
       '4.14175,-0.40982', '1.36803,-0.0982', '2.72367,-0.15974',
       '4.06735,-0.18929', '1.34368,-0.0296', '2.68647,-0.0408',
       '4.03077,-0.0275', '1.34429,0.0133', '2.6901,0.0513',
       '4.0398,0.12006', '1.3497,0.0688', '2.7033,0.16849',
       '4.0632,0.30536', '1.35989,0.13687', '2.72608,0.31093',
       '4.10095,0.52846', '1.37488,0.21753', '2.75844,0.47854',
       '4.15308,0.78933', '1.39464,0.31078', '2.80037,0.67134',
       '4.21956,1.08797', '1.4192,0.41663', '2.85187,0.88933',
       '4.30041,1.42439', '1.44853,0.53507', '2.91294,1.1325',
       '4.39561,1.7986', '1.48266,0.6661', '2.98359,1.40086',
       '4.50518,2.21058', '1.52158,0.80972', '3.06382,1.6944',
       '4.62911,2.66033', '1.56528,0.96594', '3.15362,2.01313',
       '4.7674,3.14788', '1.61378,1.13474', '3.253,2.35704',
       '4.92005,3.67318', '1.66706,1.31615', '3.36195,2.72614',
       '5.08707,4.23628', '1.72513,1.51014', '3.48048,3.12042',
       '5.26845,4.83715', '1.78798,1.71673', '3.60858,3.53989',
       '5.46419,5.4758', '1.85562,1.9359', '3.74625,3.98454',
       '5.67429,6.15222', '1.92804,2.16768', '3.81802,4.43565',
       '5.67135,6.80401', '1.85334,2.36836', '3.67004,4.83711',
       '5.45152,7.40635', '1.78149,2.56925', '3.52776,5.23898',
       '5.24025,8.0093', '1.71249,2.77032', '3.39119,5.64123',
       '5.03753,8.61283', '1.64635,2.9716', '3.26033,6.04388',
       '4.84339,9.21695', '1.58305,3.17307', '3.13518,6.44693',
       '4.6578,9.82166', '1.52262,3.37474', '3.01573,6.85036',
       '4.48077,10.42697', '1.46504,3.5766', '2.902,7.25419',
       '4.31231,11.03286', '1.41031,3.77866', '2.79397,7.65841',
       '4.15241,11.63934', '1.35844,3.98092', '2.69165,8.06303',
       '4.00107,12.24641', '1.30942,4.18338', '2.59504,8.46804',
       '3.85829,12.85407', '1.26326,4.38604', '2.50414,8.87345',
       '3.72408,13.46233', '1.21995,4.58889', '2.41895,9.27925',
       '3.59843,14.07118', '1.17949,4.79193', '2.33946,9.68543',
       '3.48134,14.6806', '1.14188,4.99518', '2.26567,10.09202',
       '3.37281,15.29063', '1.10713,5.19862', '2.1976,10.499',
       '3.27284,15.90125', '1.07524,5.40225', '2.12863,11.42365',
       '3.15871,17.96569', '1.03007,6.54205', '2.03684,13.60475',
       '3.01881,21.0896', '0.98198,7.48485', '1.93917,15.39186',
       '2.87011,23.62252', '0.93094,8.23067', '1.83562,16.78499',
       '2.71258,25.56448', '0.87696,8.77948', '1.7262,17.78413',
       '2.54625,26.91545', '0.82004,9.13132', '1.61089,18.3893',
       '2.37108,27.67545', '0.76019,9.28616', '1.48971,18.60049',
       '2.18711,27.84449', '0.69739,9.244', '1.36266,18.41768',
       '1.99432,27.42253', '0.63166,9.00486', '1.22972,17.8409',
       '1.79271,26.40962', '0.56299,8.56872', '1.09091,16.87013',
       '1.58229,24.80572', '0.49138,7.93559', '0.94622,15.50537',
       '1.36305,22.61084', '0.41683,7.10548', '0.79565,13.74664',
       '1.13499,19.825', '0.33934,6.07837', '0.6392,11.59392',
       '0.89812,16.44818', '0.25891,4.85426', '0.47688,9.04722',
       '0.65243,12.48038', '0.17555,3.43317', '0.30868,6.10654',
       '0.39792,7.92162', '0.0892,1.81507', '0.1346,2.77186',
       '0.1346,2.77186', '0,0', '0.0464,-0.99392', '0.1379,-2.87794',
       '0.0915,-1.88402', '0.22825,-4.65814', '0.40895,-8.21855',
       '0.1807,-3.56041', '0.40538,-7.90711', '0.67286,-12.93627',
       '0.26748,-5.02917', '0.57775,-10.7408', '0.92964,-17.03109',
       '0.35188,-6.29028', '0.74537,-13.15922', '1.17927,-20.50299',
       '0.43391,-7.34378', '0.90823,-15.16239', '1.42178,-23.35202',
       '0.51355,-8.18963', '1.06633,-16.75028', '1.65715,-25.57813',
       '0.59082,-8.82785', '1.21967,-17.9229', '1.88538,-27.18134',
       '0.66571,-9.25844', '1.36826,-18.68027', '2.10648,-28.16167',
       '0.73822,-9.48139', '1.51209,-19.02236', '2.32044,-28.51908',
       '0.80835,-9.49671', '1.65117,-18.94919', '2.52727,-28.25359',
       '0.8761,-9.30441', '1.78548,-18.46075', '2.72695,-27.36522',
       '0.94148,-8.90446', '1.91504,-17.55705', '2.91951,-25.85393',
       '1.00447,-8.29689', '2.03985,-16.23808', '3.10493,-23.71976',
       '1.06509,-7.48167', '2.15989,-14.50383', '3.28322,-20.96267',
       '1.12332,-6.45883', '2.27517,-12.35433', '3.45436,-17.58269',
       '1.17919,-5.22836', '2.34002,-10.3069', '3.48698,-15.24246',
       '1.14696,-4.93556', '2.28005,-9.72815', '3.40374,-14.38459',
       '1.12369,-4.65644', '2.23799,-9.17674', '3.34737,-13.56774',
       '1.10939,-4.39099', '2.21385,-8.65269', '3.31788,-12.79191',
       '1.10403,-4.13922', '2.20762,-8.15598', '3.31525,-12.0571',
       '1.10764,-3.90112', '2.21931,-7.68661', '3.33951,-11.36331',
       '1.12019,-3.6767', '2.24891,-7.2446', '3.39062,-10.71054',
       '1.14171,-3.46595', '2.29643,-6.82994', '3.46861,-10.0988',
       '1.17219,-3.26887', '2.36186,-6.44261', '3.57348,-9.52807',
       '1.21163,-3.08546', '2.44521,-6.08264', '3.70522,-8.99836',
       '1.26002,-2.91573', '2.54646,-5.75001', '3.86383,-8.50968',
       '1.31736,-2.75967', '2.66563,-5.44473', '4.0493,-8.06201',
       '1.38367,-2.61729', '2.80273,-5.1668', '4.26166,-7.65537',
       '1.45893,-2.48857', '2.95773,-4.91621', '4.50088,-7.28974',
       '1.54315,-2.37354', '3.13065,-4.69297', '4.76698,-6.96514',
       '1.63633,-2.27217', '3.32149,-4.49708', '5.05995,-6.68156',
       '1.73846,-2.18448', '3.44634,-4.28706', '5.12832,-6.30959',
       '1.68198,-2.02253', '3.33806,-3.96502', '4.97293,-5.82931',
       '1.63486,-1.8643', '3.24852,-3.6504', '4.84566,-5.36016',
       '1.59714,-1.70976', '3.17776,-3.34319', '4.74654,-4.90212',
       '1.56878,-1.55894', '3.12573,-3.04339', '4.67554,-4.45521',
       '1.5498,-1.41182', '3.09246,-2.75101', '4.63266,-4.01942',
       '1.54021,-1.26841', '3.07795,-2.46604', '4.61793,-3.59475',
       '1.53998,-1.1287', '3.08219,-2.18849', '4.63132,-3.1812',
       '1.54913,-0.99271', '3.10518,-1.91835', '4.67284,-2.77877',
       '1.56765,-0.86043', '3.14692,-1.65563', '4.74248,-2.38748',
       '1.59557,-0.73184', '3.20742,-1.40032', '4.84027,-2.00729',
       '1.63284,-0.60696', '3.28667,-1.15242', '4.96618,-1.63823',
       '1.6795,-0.4858', '3.38467,-0.91195', '5.12021,-1.28029',
       '1.73554,-0.36834', '3.50144,-0.67888', '5.30239,-0.93348',
       '1.80095,-0.25459', '3.63695,-0.45323', '5.51268,-0.59778',
       '1.87574,-0.14455', '3.79122,-0.235', '5.75112,-0.27321',
       '1.9599,-0.0382', '3.87386,-0.0202', '5.74696,0.0554',
       '1.8731,0.0756', '3.70533,0.20886', '5.50179,0.40121',
       '1.79646,0.19234', '3.55713,0.44379', '5.28711,0.75579',
       '1.72998,0.312', '3.42926,0.68456', '5.10292,1.11914',
       '1.67366,0.43457', '3.3217,0.93116', '4.94921,1.49123',
       '1.62751,0.56006', '3.23449,1.1836', '4.82601,1.87208',
       '1.59152,0.68847', '3.16759,1.44188', '4.73328,2.26167',
       '1.5657,0.8198', '3.12102,1.70599', '4.67105,2.66003',
       '1.55004,0.95404', '3.09478,1.97593', '4.63932,3.06714',
       '1.54453,1.0912', '3.08885,2.25171', '4.63805,3.48298',
       '1.5492,1.23128', '3.10327,2.53333', '4.6673,3.9076',
       '1.56402,1.37428', '3.13801,2.82078', '4.72702,4.34096',
       '1.58902,1.52019', '3.19307,3.11406', '4.81724,4.78307',
       '1.62417,1.66902', '3.26846,3.41318', '4.93795,5.23394',
       '1.66949,1.82077', '3.36418,3.71813', '5.08915,5.69356',
       '1.72497,1.97543', '3.48022,4.02892', '5.27083,6.16193',
       '1.79061,2.13301', '3.5669,4.46871', '5.32565,6.98395',
       '1.75876,2.51525', '3.49998,5.21004', '5.22045,8.06126',
       '1.72048,2.85121', '3.42022,5.85883', '5.09602,8.99974',
       '1.67579,3.1409', '3.32763,6.41509', '4.95233,9.79942',
       '1.62469,3.38433', '3.22223,6.87881', '4.7894,10.46029',
       '1.56718,3.58149', '3.104,7.24999', '4.60725,10.98236',
       '1.50325,3.73237', '2.97294,7.52861', '4.40584,11.3656',
       '1.43291,3.83698', '2.82905,7.7147', '4.18521,11.61003',
       '1.35615,3.89533', '2.67233,7.80826', '3.94531,11.71567',
       '1.27299,3.9074', '2.50279,7.80927', '3.6862,11.68248',
       '1.18341,3.87321', '2.32042,7.71775', '3.40783,11.51049',
       '1.08741,3.79274', '2.12522,7.53368', '3.11023,11.19968',
       '0.985,3.66601', '1.9172,7.25707', '2.79338,10.75007',
       '0.87618,3.493', '1.69635,6.88793', '2.45729,10.16165',
       '0.76095,3.27372', '1.46268,6.42623', '2.10197,9.43441',
       '0.6393,3.00817', '1.21617,5.872', '1.72741,8.56836',
       '0.51124,2.69636', '1.00867,5.38474', '1.49203,8.05562',
       '0.48336,2.67088', '0.95264,5.32425', '1.40758,7.95058',
       '0.45494,2.62633', '0.89554,5.22562', '1.32151,7.78832',
       '0.42598,2.56271', '0.83734,5.08883', '1.23381,7.56884',
       '0.39646,2.48001', '0.77804,4.91391', '1.14446,7.29215',
       '0.36642,2.37824', '0.71768,4.70083', '1.0535,6.95823',
       '0.33583,2.2574', '0.65622,4.44962', '0.96091,6.5671',
       '0.30469,2.11749', '0.59367,4.16025', '0.86668,6.11876',
       '0.273,1.9585', '0.53004,3.83274', '0.77082,5.61318',
       '0.24078,1.78044', '0.46532,3.46709', '0.67333,5.0504',
       '0.20802,1.58331', '0.39952,3.06329', '0.57422,4.4304',
       '0.17471,1.36711', '0.33262,2.62135', '0.47347,3.75318',
       '0.14085,1.13183', '0.26464,2.14126', '0.37109,3.01874',
       '0.10646,0.87748', '0.19557,1.62302', '0.26709,2.22708',
       '0.0715,0.60406', '0.12541,1.06664', '0.16144,1.37821',
       '0.036,0.31157', '0.0542,0.47212', '0.0542,0.47212']

    #points = points.split(' ')
    points = [tuple([float(x) for x in point.split(',')]) for point in points]

    vertices = [[xstart, ystart]]
    codes = [path.Path.MOVETO]

    coords = []

    while len(points) > 0:
        ref_x, ref_y = vertices[-1][0], vertices[-1][1]

        x1, y1 = points.pop(0)
        x2, y2 = points.pop(0)
        x, y = points.pop(0)

        codes += [path.Path.CURVE4] * 3
        vertices.append([x1 + ref_x, y1 + ref_y])
        vertices.append([x2 + ref_x, y2 + ref_y])
        vertices.append([x + ref_x, y + ref_y])

        coords.append([x, y])

    vertices = numpy.array(vertices)
    vertices[:, 0] = (vertices[:, 0] - xn90) / xscale - 90
    vertices[:, 1] = (vertices[:, 1] - y0) / yscale

    return vertices, codes


def getAngles(amin, amax, n=300):
    # angles for tomography
    angles = numpy.arange(n + 1) / float(n) * (amax - amin) + amin
    angles_radian = angles * 2 * numpy.pi / 360
    return angles, angles_radian


def calculateSqueezing(angles_radian, sy, sz):
    # Calculate \Delta^2 \hat{S}_\theta.
    # sy, sz and the result have shape (subsets, points)
    N = 1200.0
    ss = sy.shape[0]
    tp = angles_radian.size
    ca = numpy.tile(numpy.cos(angles_radian), (ss, 1))
    sa = numpy.tile(numpy.sin(angles_radian), (ss, 1))

    # calculate mean inside subset and tile it to match cosine and sine data
    mean = lambda x: numpy.tile(x.mean(1).reshape(ss, 1), (1, tp))

    d2S = mean(sz ** 2) * ca ** 2 + mean(sy ** 2) * sa ** 2 - \
        2 * mean(sz * sy) * sa * ca - mean(sy) ** 2 * sa ** 2 - mean(sz) ** 2 * ca ** 2 + \
        2 * mean(sz) * mean(sy) * sa * ca
    return d2S / N * 4


def riedel_rotation(fname):

    with open(get_path(__file__, 'split_potentials_spins_last.pickle'), 'rb') as f:
        spins = pickle.load(f)

    Sx = spins['Sx']
    # for some reason Y direction in Riedel is swapped (is it the sign of detuning?)
    Sy = -spins['Sy']
    Sz = spins['Sz']

    amin = -90
    amax = 90
    angles, angles_radian = getAngles(amin, amax)

    ens = Sy.size # total number of ensembles
    # average result using all data
    res_full = calculateSqueezing(angles_radian, Sy.reshape(1, ens), Sz.reshape(1, ens))

    vertices, codes = buildRiedelTomographyPath()
    riedel_path = path.Path(vertices, codes)
    #patch = patches.PathPatch(riedel_path, edgecolor=mplh.color.f.blue.main,
    #    facecolor='none', linestyle='dashdot')

    riedel_path = riedel_path.vertices
    riedel_path_x = riedel_path[:,0]
    riedel_path_y = riedel_path[:,1]

    fig = mplh.figure(width=0.75)
    subplot = fig.add_subplot(111)

    #subplot.add_patch(patch)
    subplot.plot([amin, amax], [0, 0], color='grey', linewidth=0.5,
        linestyle='-.', dashes=mplh.dash['-.'])

    subplot.plot(riedel_path_x, riedel_path_y, color=mplh.color.f.blue.main,
        linestyle='--', dashes=mplh.dash['--'])

    subplot.plot(angles, numpy.log10(res_full[0]) * 10, color=mplh.color.f.red.main)

    subplot.set_xlim(xmin=amin, xmax=amax)
    subplot.set_ylim(ymin=-13, ymax=20)
    subplot.xaxis.set_ticks((-90, -45, 0, 45, 90))
    subplot.xaxis.set_ticklabels(('$-90$', '$-45$', '$0$', '$45$', '$90$'))
    subplot.set_xlabel('$\\theta$ (degrees)')
    subplot.set_ylabel('$N \\Delta \\hat{S}_\\theta^2 / \\langle \\hat{\\mathbf{S}} \\rangle^2$ (dB)')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def getHeightmap(X, Y, xmin, xmax, ymin, ymax, xbins, ybins, cloud_levels):
    """Returns heightmap, extent and levels for contour plot"""

    hist, xedges, yedges = numpy.histogram2d(X, Y, bins=(xbins, ybins),
        range=[[xmin, xmax], [ymin, ymax]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    hmax = hist.max()
    levels = numpy.exp(numpy.arange(cloud_levels + 1) / float(cloud_levels) * numpy.log(hmax))
    return hist.T, extent, levels


def riedel_cloud(fname):

    cloud_xsize = 100.0
    cloud_ysize = 250.0
    cloud_zsize = 50.0
    cloud_zbins = 25
    cloud_levels = 20

    cloud_ybins = int(float(cloud_zbins) * cloud_ysize / cloud_zsize + 0.5)
    cloud_xbins = int(cloud_zbins * cloud_xsize / cloud_zsize + 0.5)
    cloud_ybins = int(cloud_zbins * cloud_ysize / cloud_zsize + 0.5)

    with open(get_path(__file__, 'split_potentials_spins_last.pickle'), 'rb') as f:
        spins = pickle.load(f)

    Sx = spins['Sx']
    # for some reason Y direction in Riedel is swapped (is it the sign of detuning?)
    Sy = -spins['Sy']
    Sz = spins['Sz']

    Sx -= Sx.mean()
    Sy -= Sy.mean()
    Sz -= Sz.mean()

    nullfmt = NullFormatter()
    fig_width = 8

    x_d = 0.09
    x_dd = 0.03
    x_ly = (1.0 - x_d - x_dd * 2) / (1.0 + cloud_zsize / cloud_xsize)
    x_lx = (1.0 - x_d - x_dd * 2) / (1.0 + cloud_xsize / cloud_zsize)

    y_d = x_d
    y_dd = x_dd
    y_lx = x_lx
    y_lz = y_lx / cloud_xsize * cloud_zsize

    aspect = (x_d + x_dd * 2 + x_ly + x_lx) / (y_d + y_dd * 2 + y_lx + y_lz)
    y_d *= aspect
    y_dd *= aspect
    y_lx *= aspect
    y_lz *= aspect

    # definitions for the axes
    rectYZ = [x_d, y_d, x_ly, y_lz]
    rectXY = [x_d, y_d + y_dd + y_lz, x_ly, y_lx]
    rectXZ = [x_d + x_dd + x_ly, y_d, x_lx, y_lz]

    # start with a rectangular Figure
    fig = mplh.figure(width=1., aspect=1./aspect)

    axYZ = plt.axes(rectYZ)
    axXY = plt.axes(rectXY)
    axXZ = plt.axes(rectXZ)

    # no labels
    axXY.xaxis.set_major_formatter(nullfmt)
    axXZ.yaxis.set_major_formatter(nullfmt)

    hm, extent, levels = getHeightmap(
        Sy, Sz, -cloud_ysize, cloud_ysize, -cloud_zsize, cloud_zsize, cloud_ybins, cloud_zbins,
        cloud_levels)
    axYZ.contourf(hm, extent=extent, cmap=mplh.cm_zeropos, levels=levels)
    axYZ.set_xlabel('$S_y$')
    axYZ.set_ylabel('$S_z$')
    axYZ.set_xlim(xmin=-cloud_ysize, xmax=cloud_ysize)
    axYZ.set_ylim(ymin=-cloud_zsize, ymax=cloud_zsize)

    hm, extent, levels = getHeightmap(
        Sy, Sx, -cloud_ysize, cloud_ysize, -cloud_xsize, cloud_xsize, cloud_ybins, cloud_xbins,
        cloud_levels)
    axXY.contourf(hm, extent=extent, cmap=mplh.cm_zeropos, levels=levels)
    axXY.set_ylabel('$S_x$')
    axXY.set_xlim(xmin=-cloud_ysize, xmax=cloud_ysize)
    axXY.set_ylim(ymin=-cloud_xsize, ymax=cloud_xsize)
    axXY.yaxis.set_ticks((-80, -40, 0, 40, 80))

    hm, extent, levels = getHeightmap(
        Sx, Sz, -cloud_xsize, cloud_xsize, -cloud_zsize, cloud_zsize, cloud_xbins, cloud_zbins,
        cloud_levels)
    axXZ.contourf(hm, extent=extent, cmap=mplh.cm_zeropos, levels=levels)
    axXZ.set_xlabel('$S_x$')
    axXZ.set_xlim(xmin=-cloud_xsize, xmax=cloud_xsize)
    axXZ.set_ylim(ymin=-cloud_zsize, ymax=cloud_zsize)


    # Plot the arrows on the YZ facet

    # optimal squeezing angle and corresponding variance, to plot supporting info
    min_angle = 9.6 / 180 * numpy.pi
    min_var = numpy.sqrt(25.349280804)

    # parameters for arrow pointing at the best squeezing
    arrow_len = 30
    arrow1_x = -(min_var + arrow_len) * numpy.sin(min_angle)
    arrow1_y = (min_var + arrow_len) * numpy.cos(min_angle)
    arrow1_dx = (arrow_len) * numpy.sin(min_angle)
    arrow1_dy = -(arrow_len) * numpy.cos(min_angle)

    arrow_kwds = dict(
        width=2,
        head_width=5,
        linewidth=0.5,
        shape="full",
        overhang=0,
        head_starts_at_zero=False,
        fill=False,
        length_includes_head=True,
        facecolor=mplh.color.f.blue.main)

    # supporting lines

    r = numpy.array([0, 1.0])

    # axis of the ellipse
    l1_x = (r * 2 - 1) * min_var * numpy.sin(min_angle)
    l1_y = (-r * 2 + 1) * min_var * numpy.cos(min_angle)

    # horizontal line
    l2_x = r * 150
    l2_y = r * 0

    # projection direction line
    l3_x = r * 150
    l3_y = r * 150 * numpy.sin(min_angle)

    # plot pointing arrows
    axYZ.arrow(arrow1_x, arrow1_y, arrow1_dx, arrow1_dy, **arrow_kwds)
    axYZ.arrow(-arrow1_x, -arrow1_y, -arrow1_dx, -arrow1_dy, **arrow_kwds)

    # plot supporting lines
    #axYZ.plot(l1_x, l1_y, color=mplh.color.f.blue.main, linewidth=0.5)
    axYZ.plot(l2_x, l2_y, linestyle='--', color='black', linewidth=0.5, dashes=mplh.dash['--'])
    axYZ.plot(l3_x, l3_y, linestyle='--', color='black', linewidth=0.5, dashes=mplh.dash['--'])

    # mark angle
    arc = patches.Arc((0.0, 0.0), 100, 100,
        theta1=0, theta2=min_angle / numpy.pi * 180, linewidth=0.5, fill=False)
    axYZ.add_patch(arc)

    # plot labels
    axYZ.text(-30, 20, "$d_\\theta$")
    axYZ.text(40, 10, "$\\theta$")


    fig.savefig(fname)


def _feshbach_squeezing(fname, losses):

    with open(get_path(__file__, 'feshbach_squeezing' + ('' if losses else '_no_losses') + '.json')) as f:
        sq = json.load(f)

    t_sq = numpy.array(sq['times'])
    xi2_sq_80 = numpy.array(sq['xi2_80.0'])
    xi2_sq_85 = numpy.array(sq['xi2_85.0'])
    xi2_sq_90 = numpy.array(sq['xi2_90.0'])
    xi2_sq_95 = numpy.array(sq['xi2_95.0'])

    fig = mplh.figure(width=0.5)
    subplot = fig.add_subplot(111)

    subplot.plot(t_sq * 1e3, 10 * numpy.log10(xi2_sq_80), color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])
    subplot.plot(t_sq * 1e3, 10 * numpy.log10(xi2_sq_85), color=mplh.color.f.red.main,
        linestyle='--', dashes=mplh.dash['--'])
    subplot.plot(t_sq * 1e3, 10 * numpy.log10(xi2_sq_90), color=mplh.color.f.green.main,
        linestyle=':', dashes=mplh.dash[':'])
    subplot.plot(t_sq * 1e3, 10 * numpy.log10(xi2_sq_95), color=mplh.color.f.yellow.main,
        linestyle='-.', dashes=mplh.dash['-.'])

    subplot.plot([0, 100], [0, 0], color='grey', linewidth=0.5,
        linestyle='-.', dashes=mplh.dash['-.'])

    subplot.set_xlim(xmin=0, xmax=100)
    subplot.set_ylim(ymin=-13 if losses else -20 , ymax=1)
    subplot.set_xlabel('$T$ (ms)')
    subplot.set_ylabel('$\\xi^2$ (dB)')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)

def feshbach_squeezing(fname):
    _feshbach_squeezing(fname, True)

def feshbach_squeezing_no_losses(fname):
    _feshbach_squeezing(fname, False)


def feshbach_scattering(fname):

    B0 = 9.1047
    DB = 2e-3
    gB = 4.7e-3
    a_bg = 97.7

    fig = mplh.figure(width=0.75)
    subplot = fig.add_subplot(111)

    B = lambda x: B0 + gB * x
    fa = lambda x: (1 - DB / (B(x) - B0 - 1j * gB / 2))
    x = numpy.linspace(-1, 16, 200)

    #subplot.plot([0, 0], [0, 2], color='grey', linewidth=0.5,
    #    linestyle='-.', dashes=mplh.dash['-.'])

    subplot.plot(x, -fa(x).imag, color=mplh.color.f.red.main,
        linestyle='--', dashes=mplh.dash['--'])
    subplot.plot(x, fa(x).real, color=mplh.color.f.blue.main)

    a12s = (80., 85., 90., 95.)
    diffs = []
    for a12 in a12s:
        hbar = 1.054571628e-34
        r_bohr = 5.2917720859e-11
        m = 1.443160648e-25

        a = 1 - a12 / a_bg
        xx = (DB / a + numpy.sqrt((DB / a) ** 2 - gB ** 2)) / (2 * gB)
        print xx * dB, a_bg * fa(xx).real, -fa(xx).imag * a_bg * r_bohr * 8 * numpy.pi * hbar / m
        diffs.append(xx)

    #for xx in (0.5, 0.75, 1.0, 1.5):
    for xx, a12 in zip(diffs, a12s):

        subplot.plot([xx, xx], [0, fa(xx).real], color='grey', linewidth=0.5,
            linestyle='-.', dashes=mplh.dash['-.'])
        subplot.scatter([xx], [fa(xx).real], color=mplh.color.f.blue.darkest, marker='.')
        subplot.scatter([xx], [-fa(xx).imag], color=mplh.color.f.red.darkest, marker='.')
        subplot.text(
            xx - (0.9 if a12 > 80 else 1.3),
            fa(xx).real + 0.07,
            "$" + str(int(a12)) + "\\,r_B$")

    subplot.set_xlim(xmin=-1, xmax=16)
    subplot.set_ylim(ymin=0, ymax=1.5)
    subplot.set_xlabel('$(B - B_0) / \\gamma_B$')
    subplot.text(-0.1, 1.35, "$\\mathrm{Re}$")
    subplot.text(-0.5, 0.2, "$-\\mathrm{Im}$")
    subplot.set_ylabel('$a(B)/a_{\\mathrm{bg}}$')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)
