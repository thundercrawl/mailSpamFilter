def score_model(path, uri, port):
    """
    Score mailfile on the local path with MLflow model deployed at given uri and port.

    :param path: Path to a single mail file.
    :param uri: URI the model is deployed at
    :param port: Port the model is deployed at.
    :return: Server response.
    """
    text = "path:" + path + " uri:" + uri + " port:" + port
    savelog_to_file(text,False)
    
    file_c = ""
    with open(path) as f:
        for line in f:
            strall = line.strip()
            file_c = file_c + strall + " "

    dictionary = read_dic(directory_file)
    features_test = build_feature_for_String(file_c,dictionary)
    curstr = "{\"columns\":["
    columns = ""
    for i in dictionary:
        columns = columns + "\"" + i + "\","

    curstr = curstr + columns[0:-1] + "],"

    dic={}
    dic['data']=features_test.tolist()
    dicJson = json.dumps(dic)
    json1 = dicJson.replace("{","")
    json2 = json1.replace("}","")
    curstr = curstr + json2 + "}"

    savejson_to_file(path[path.rfind("/") + 1:] + ".log",curstr)
    # data = pd.DataFrame(data=[base64.encodebytes(read_image(x)) for x in filenames],
    #                     columns=["image"]).to_json(orient="split")

    response = requests.post(url='{uri}:{port}/invocations'.format(uri=uri, port=port),
                             data=curstr,
                             headers={"Content-Type": "application/json; format=pandas-split"})

    if response.status_code != 200:
        raise Exception("Status Code {status_code}. {text}".format(
            status_code=response.status_code,
            text=response.text
        ))
    return response