import argparse
import json
import sys
import os

from pathlib import Path

from utils.core import main


def arg_parameter(arr):
    # 參數初始化設定
    arg = argparse.ArgumentParser()

    for key in arr:
        a = arr[key]
        arg.add_argument("--" + a["name"], type=a["type"], default=a["default"])

    # 讀入參數
    return arg.parse_args()


def report_process(anomaly_dict):
    text = """Tx:Check Panoramic radiography films for initial examination in this year
    C.F.:
    1.缺牙: %(missing)s
    2.殘根: %(R.R)s  埋伏齒: %(embedded)s
    3.固定補綴物: %(filling)s
    4.齲齒: %(caries)s
    5.曾經根管治療: %(endo)s"""

    predict = {}
    anomaly_list = ['R.R', 'caries', 'crown', 'endo', 'post', 'filling', 'Imp', 'embedded', 'impacted', 'missing']
    for filename, teeth in anomaly_dict.items():
        filename = f'{filename}.jpg'
        predict[filename] = {}

        teeth_anomalies_dict = {anomaly: [] for anomaly in anomaly_list}
        for tooth_number, anomalies in teeth.items():
            for anomaly in anomalies:
                teeth_anomalies_dict[anomaly].append(tooth_number)

        text_anomalies_list = ['missing', 'R.R', 'embedded', 'filling', 'caries', 'endo']
        text_dict = {
            anomaly: ' '.join(map(str, teeth_anomalies_dict[anomaly])) if teeth_anomalies_dict[
                anomaly] else 'no finding' for
            anomaly in text_anomalies_list}

        predict[filename]['text'] = text % text_dict
        predict[filename]['data'] = teeth_anomalies_dict

    return {
        "isSuccessful": True,
        "msg": "辨識完成",
        "predict": predict,
        "dir": para.Dir,  # 測試用請刪除
    }


if __name__ == '__main__':
    # cmd參數初始化
    args = {
        0: {"name": "Dir", "type": str, "default": "test"},
    }
    # 取得cmd參數
    para = arg_parameter(args)
    data_dir = Path(para.Dir) / 'test1'
    image_names = list(data_dir.glob('*.jpg'))

    tooth_anomaly_dict = main(image_names)

    result = report_process(tooth_anomaly_dict)

    print(json.dumps(result, ensure_ascii=False))
