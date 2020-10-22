old_dashboard_name = "old_dashboard"
old_dashboard_state = {
    "charts": {
        "1": {
            "autoSize": True,
            "areFiltersInverse": False,
            "cap": 12,
            "renderArea": False,
            "color": {
                "type": "none"
            },
            "colorDomain": None,
            "dcFlag": 1,
            "densityAccumulatorEnabled": True,
            "dimensions": [
                {
                    "inactive": False,
                    "name": None,
                    "isRequired": False,
                    "isError": False,
                    "table": "test_data_no_nulls_ipc",
                    "type": "STR",
                    "precision": 0,
                    "is_array": False,
                    "is_dict": True,
                    "name_is_ambiguous": False,
                    "label": "text_",
                    "value": "text_",
                    "custom": False,
                    "axisLabel": None,
                    "max_val": None,
                    "min_val": None,
                    "maxBinSize": None,
                    "currentHighValue": None,
                    "currentLowValue": None,
                    "autobin": False,
                    "numOfBins": None,
                    "isBinned": False,
                    "isBinnable": False
                },
                {
                    "isRequired": False,
                    "isError": False
                }
            ],
            "elasticX": True,
            "elasticY": True,
            "filters": [],
            "geoJson": None,
            "loading": True,
            "measures": [
                {
                    "inactive": False,
                    "name": "col0",
                    "isRequired": False,
                    "isError": False,
                    "label": "# Records",
                    "value": "*",
                    "type": "SMALLINT",
                    "custom": False,
                    "axisLabel": None,
                    "minMax": None,
                    "categories": None,
                    "colorType": "quantitative",
                    "aggType": "Count",
                    "originIndex": 0
                },
                {
                    "isRequired": False,
                    "isError": False
                }
            ],
            "rangeChartEnabled": False,
            "rangeFilter": [],
            "savedColors": {},
            "sortColumn": None,
            "ticks": 3,
            "title": "",
            "type": "table",
            "showOther": False,
            "showNullDimensions": True,
            "markTypes": [],
            "multiSources": {},
            "hasError": False,
            "dataSource": "test_data_no_nulls_ipc",
            "hoverSelectedColumns": []
        },
        "test_data_no_nulls_ipc": {
            "dcFlag": 2,
            "loading": False,
            "color": {
                "defaultOtherDomain": "Default"
            }
        }
    },
    "ui": {
        "showFilterPanel": False,
        "showClearFiltersDropdown": False,
        "banner": {
            "open": False,
            "type": None
        },
        "modal": {
            "type": None,
            "open": False,
            "content": "",
            "header": "",
            "primaryAction": {
                "text": "OK"
            },
            "secondaryAction": {
                "text": "CANCEL"
            }
        },
        "selectorPillHover": {
            "shouldShowPrompt": False,
            "message": "",
            "top": 0
        },
        "selectorPositions": {
            "dimensions": [
                202.60000610351562,
                242.60000610351562,
                202.60000610351562,
                242.60000610351562,
                202.60000610351562
            ],
            "measures": [
                331.3999938964844,
                371.3999938964844,
                331.3999938964844,
                291.3999938964844
            ]
        },
        "nagScreenIsEnabled": False
    },
    "filters": [],
    "dashboard": {
        "id": None,
        "title": "old_dashboard_2",
        "chartContainers": [
            {
                "id": "1"
            }
        ],
        "table": None,
        "filtersId": [],
        "layout": [
            {
                "w": 10,
                "h": 10,
                "x": 0,
                "y": 0,
                "i": "1",
                "minW": 5,
                "minH": 5,
                "moved": False,
                "static": False
            }
        ],
        "currentDataSource": "test_data_no_nulls_ipc",
        "dataSources": {
            "test_data_no_nulls_ipc": {
                "alias": "A",
                "columnMetadata": [
                    {
                        "table": "test_data_no_nulls_ipc",
                        "type": "TINYINT",
                        "precision": 0,
                        "is_array": False,
                        "is_dict": False,
                        "name_is_ambiguous": False,
                        "label": "tinyint_",
                        "value": "tinyint_"
                    },
                    {
                        "table": "test_data_no_nulls_ipc",
                        "type": "SMALLINT",
                        "precision": 0,
                        "is_array": False,
                        "is_dict": False,
                        "name_is_ambiguous": False,
                        "label": "smallint_",
                        "value": "smallint_"
                    },
                    {
                        "table": "test_data_no_nulls_ipc",
                        "type": "INT",
                        "precision": 0,
                        "is_array": False,
                        "is_dict": False,
                        "name_is_ambiguous": False,
                        "label": "int_",
                        "value": "int_"
                    },
                    {
                        "table": "test_data_no_nulls_ipc",
                        "type": "BIGINT",
                        "precision": 0,
                        "is_array": False,
                        "is_dict": False,
                        "name_is_ambiguous": False,
                        "label": "bigint_",
                        "value": "bigint_"
                    },
                    {
                        "table": "test_data_no_nulls_ipc",
                        "type": "FLOAT",
                        "precision": 0,
                        "is_array": False,
                        "is_dict": False,
                        "name_is_ambiguous": False,
                        "label": "float_",
                        "value": "float_"
                    },
                    {
                        "table": "test_data_no_nulls_ipc",
                        "type": "DOUBLE",
                        "precision": 0,
                        "is_array": False,
                        "is_dict": False,
                        "name_is_ambiguous": False,
                        "label": "double_",
                        "value": "double_"
                    },
                    {
                        "table": "test_data_no_nulls_ipc",
                        "type": "DATE",
                        "precision": 0,
                        "is_array": False,
                        "is_dict": False,
                        "name_is_ambiguous": False,
                        "label": "date_",
                        "value": "date_"
                    },
                    {
                        "table": "test_data_no_nulls_ipc",
                        "type": "TIMESTAMP",
                        "precision": 0,
                        "is_array": False,
                        "is_dict": False,
                        "name_is_ambiguous": False,
                        "label": "datetime_",
                        "value": "datetime_"
                    },
                    {
                        "table": "test_data_no_nulls_ipc",
                        "type": "TIME",
                        "precision": 0,
                        "is_array": False,
                        "is_dict": False,
                        "name_is_ambiguous": False,
                        "label": "time_",
                        "value": "time_"
                    },
                    {
                        "table": "test_data_no_nulls_ipc",
                        "type": "STR",
                        "precision": 0,
                        "is_array": False,
                        "is_dict": True,
                        "name_is_ambiguous": False,
                        "label": "text_",
                        "value": "text_"
                    }
                ]
            }
        },
        "privileges": {},
        "saveLinkState": {
            "error": False,
            "request": False,
            "saveLinkId": None
        },
        "loadState": {
            "error": False,
            "dataAccessError": False,
            "complete": True,
            "request": False,
            "loadLinkId": None
        },
        "saveState": {
            "request": False,
            "error": False,
            "isSaved": False
        },
        "copyState": {
            "error": False,
            "request": False
        },
        "streaming": {
            "interval": 0,
            "request": False
        },
        "version": "4.6.1-20190429-02ec2e206b"
    },
    "machineLearning": {
        "training": False,
        "queued": False,
        "error": None,
        "predictionCol": None,
        "featureCols": [],
        "conditions": [],
        "algorithm": "Linear Regression",
        "nModels": 3,
        "nAlphas": 4,
        "nLambdas": 100,
        "nFolds": 4,
        "tensorFlowInput1": 0,
        "tensorFlowInput2": 0,
        "deepLearningInput1": "foo",
        "tableWithPredictions": None,
        "trainingTable": None,
        "predictedColumn": None,
        "accuracyColumn": None,
        "modelName": "",
        "predicting": False,
        "models": [],
        "complete": False,
        "trainResults": None,
        "accuracy": None,
        "jobId": None,
        "progress": {
            "steps": None,
            "current": None
        }
    }
}