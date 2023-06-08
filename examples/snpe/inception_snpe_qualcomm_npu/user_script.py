# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
def post_process(output):
    return output["results"]["InceptionV3/Predictions/Reshape_1:0"].squeeze(1).argmax(axis=1)
