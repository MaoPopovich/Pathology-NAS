import xmltodict
import geojson
from geojson import Polygon, Feature, FeatureCollection
import os
import json
import xml.etree.ElementTree as ET
import copy
import numpy as np


def camelyon16xml2json(inxml, outjson):
        """
        Convert an annotation of camelyon16 xml format into a json format.
        Arguments:
            inxml: string, path to the input camelyon16 xml format
            outjson: string, path to the output json format
        """
        root = ET.parse(inxml).getroot()
        annotations_tumor = \
            root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
        annotations_0 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
        annotations_1 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
        annotations_2 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
        annotations_positive = \
            annotations_tumor + annotations_0 + annotations_1   # Tumor & 0 & 1 are positive area
        annotations_negative = annotations_2  # 2 is negative area

        json_dict = {}
        json_dict['positive'] = []
        json_dict['negative'] = []

        for annotation in annotations_positive:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['positive'].append({'name': name, 'vertices': vertices})

        for annotation in annotations_negative:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['negative'].append({'name': name, 'vertices': vertices})

        with open(outjson, 'w') as f:
            json.dump(json_dict, f, indent=1)

labels = {"Tumor":"positive", "_0": "positive", "_1":"positive", "_2":"negative", "Exclusion":"negative", "None":"None"}

def convert_xml_to_geojson(xml_path):
    with open(xml_path, 'r') as file:
        xml_string = file.read()

    # 解析XML字符串为字典
    xml_dict = xmltodict.parse(xml_string)
    # print(type(xml_dict))

    # 从字典中提取注释
    annotations = xml_dict['ASAP_Annotations']['Annotations']['Annotation']

    # if only one annotated area, convert into list
    if isinstance(annotations, dict):
        annotations = [annotations]

    # 将注释转换为GeoJSON特征
    features = []
    for annotation in annotations:
        name = annotation['@Name']
        group = annotation['@PartOfGroup']
        label = labels[group]  # positive: tumor & _0 & _1; negative:_2
        color = annotation['@Color']
        coordinates = annotation['Coordinates']['Coordinate']
        if isinstance(coordinates, dict):
            coordinates = [coordinates]
        points = [(float(coordinate['@X']), float(coordinate['@Y'])) for coordinate in coordinates]
        # 为了形成一个闭合的多边形，首尾坐标需要相同
        points.append(points[0])
        polygon = Polygon([points])
        feature = Feature(geometry=polygon, properties={"name": name, "label": label, "color": color})
        features.append(feature)

    # 创建一个GeoJSON特征集合
    feature_collection = FeatureCollection(features)

    # 将特征集合转换为GeoJSON字符串
    geojson_string = geojson.dumps(feature_collection)

    return geojson_string


# 使用
xml_dir = 'dataset/lesion_annotations_test'
geojson_dir = xml_dir.replace('lesion_annotations_test', 'wsi_label_test')
os.makedirs(geojson_dir, exist_ok=True)
for file in os.listdir(xml_dir)[47:]:
    print(file)
    out_path = os.path.join(geojson_dir, file.replace('.xml', '.geojson'))
    geojson_string = convert_xml_to_geojson(os.path.join(xml_dir, file))
    with open(out_path, 'w') as file:
        file.write(geojson_string)
    # camelyon16xml2json(os.path.join(xml_dir, file), out_path)