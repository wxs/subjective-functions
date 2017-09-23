# Copyright 2017, Xavier Snelgrove
import os
if __name__ == "__main__":
    from gram import load_model
    valid_model = construct_gatys_model(padding='valid')
    same_model = construct_gatys_model(padding='same')
    if not os.path.exists("model_data"):
        os.mkdir("model_data")
    valid_model.save("model_data/gatys_valid.h5")
    same_model.save("model_data/gatys_same.h5")
