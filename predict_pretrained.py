# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.

Skript testet das vortrainierte Modell


@author: Maurice Rohr
"""


from predict_tf import predict_labels
from wettbewerb import EEGDataset, save_predictions
import argparse
import time
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('--test_dir', action='store',type=str,default=r'D:\datasets\eeg\dataset_dir_original\shared_data\training_mini')
    # parser.add_argument('--model_name', action='store',type=str,default='model.json')
    parser.add_argument('--model_name', action='store',type=str,default='transformer_model.json')
    # parser.add_argument('--model_name', action='store',type=str,default='model_CBraMod.json')
    # parser.add_argument('--model_name', action='store',type=str,default='model_tf_optimized.json')
    parser.add_argument('--allow_fail',action='store_true',default=False)
    args = parser.parse_args()
    
    # Erstelle EEG Datensatz aus Ordner
    dataset = EEGDataset(args.test_dir)
    print(f"Teste Modell auf {len(dataset)} Aufnahmen")
    
    predictions = list()
    start_time = time.time()
    
    # Rufe Predict Methode für jedes Element (Aufnahme) aus dem Datensatz auf
    for item in tqdm(dataset, desc="Predicting", unit="sample"):
        id, channels, data, fs, ref_system, eeg_label = item
        _prediction = predict_labels(channels, data, fs, ref_system, model_name=args.model_name)
        _prediction["id"] = id
        predictions.append(_prediction)

    pred_time = time.time()-start_time
    
    save_predictions(predictions) # speichert Prädiktion in CSV Datei
    print("Runtime",pred_time,"s")