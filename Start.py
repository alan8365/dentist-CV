import argparse
import json

from utils.core import main


def arg_parameter(arr):
    # 參數初始化設定
    arg = argparse.ArgumentParser()

    for key in arr:
        a = arr[key]
        arg.add_argument("--" + a["name"], type=a["type"], default=a["default"])

    # 讀入參數
    return arg.parse_args()


if __name__ == '__main__':
    # cmd參數初始化
    args = {
        0: {"name": "Dir", "type": str, "default": ""},
    }
    # 取得cmd參數
    para = arg_parameter(args)
    # para.Dir 即可取得傳入的資料夾路徑
    # print(para.Dir)

    tooth_anomaly_dict = main(para.Dir)

    text = """Tx:Check Panoramic radiography films for initial examination in this year
    C.F.:
    1.缺牙: 47
    2.殘根: 16 25  埋伏齒: no finding
    3.固定補綴物: no finding
    4.齲齒: 17 18
    5.曾經根管治療: 15 16 24"""

    predict = {}
    anomaly_list = ['R.R', 'caries', 'crown', 'endo', 'filling', 'post', 'filling', 'Imp', 'embedded', 'impacted',
                    'missing']
    for filename, teeth in tooth_anomaly_dict.items():
        filename = f'{filename}.jpg'
        predict[filename] = {anomaly: [] for anomaly in anomaly_list}
        for tooth_number, anomalies in teeth.items():
            for anomaly in anomalies:
                predict[filename][anomaly].append(tooth_number)
    print(predict)

    # predict = {
    #     "img01.jpg": {
    #         "text": text,
    #         "data": {
    #             "caries": [17, 18],
    #             "endo": [15, 16, 24],
    #             "R.R": [16, 25],
    #         }
    #     },
    #     "img02.jpg": {
    #         "text": text,
    #         "data": {
    #             "caries": [17, 18],
    #             "endo": [15, 16, 24],
    #             "R.R": [16, 25],
    #         }
    #     }
    # }

    result = {
        "isSuccessful": True,
        "msg": "辨識完成",
        "predict": predict,
        "dir": para.Dir,  # 測試用請刪除
    }

    print(json.dumps(result, ensure_ascii=False))
