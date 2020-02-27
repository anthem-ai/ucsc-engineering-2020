import glob
import pandas as pd
patients = {}

for file in glob.glob('json/*.json'):
    data = pd.read_json(file)

    # Categories of the patient information
    patients[data['entry'][0]['resource']['id']] = {}
    patients[data['entry'][0]['resource']['id']]['Body Mass Index'] = {}
    patients[data['entry'][0]['resource']['id']]['Hospitalized'] = 0
    patients[data['entry'][0]['resource']['id']]['Age'] = 0
    patients[data['entry'][0]['resource']['id']]['Smoking'] = 0

    for i in range(data['entry'].count()):

        # Check the age of the patient
        if 'birthDate' in data['entry'][0]['resource']:
            age = 2020 - (int)(data['entry'][0]['resource']['birthDate'][0:4])
            patients[data['entry'][0]['resource']['id']]['Age'] = age

        # Check if Patient has been Hospitalized (No date restrictions currently)
        if (data['entry'][i]['resource']['resourceType'] == "Encounter"):
            if (data['entry'][i]['resource']['class']['code'] == "IMP"):
                patients[data['entry'][0]['resource']['id']]['Hospitalized'] = 1
            if (data['entry'][i]['resource']['class']['code'] == "EMER"):
                patients[data['entry'][0]['resource']['id']]['Hospitalized'] = 1

        if(data['entry'][i]['resource']['resourceType'] == "Observation"):

            # Body Mass Readings, could be replaced with other observations or more generalized code status
            if (data['entry'][i]['resource']['code']['text'] == "Body Mass Index"):
                date = data['entry'][i]['resource']['effectiveDateTime'][0:-15]
                value = data['entry'][i]['resource']['valueQuantity']['value']
                patients[data['entry'][0]['resource']['id']]['Body Mass Index'][date] = value
                
            # Smoking Status
            if(data['entry'][i]['resource']['code']['text'] == "Tobacco smoking status NHIS"):
                if (data['entry'][i]['resource']['valueCodeableConcept']['coding'][0]['display'] == "Former smoker"):
                    patients[data['entry'][0]['resource']['id']]['Smoking'] = 1
                if (data['entry'][i]['resource']['valueCodeableConcept']['coding'][0]['display'] == "Never smoker"):
                    patients[data['entry'][0]['resource']['id']]['Smoking'] = 0
                break

# Test printing code
'''
    #for i in range(data['entry'].count()):
        #if(data['entry'][i]['resource']['resourceType'] == "Observation"):
            #if (data['entry'][i]['resource']['code']['text'] == "Body Mass Index"):
                #print("BMI = ", data['entry'][i]['resource']['valueQuantity']['value'])
                
        #Check if Patient has been Hospitalized
        if (data['entry'][i]['resource']['class']['code']):
            print(data['entry'][i]['resource']['class']['code'])

            if(data['entry'][i]['resource']['class']['code'] == "IMP"):
                patients[data['entry'][0]['resource']['id']]['Hospitalized'] = 1
            if(data['entry'][i]['resource']['class']['code'] == "EMER"):
                patients[data['entry'][0]['resource']['id']]['Hospitalized'] = 1

if 'birthDate' in data['entry'][0]['resource']:
    age = 2020 - (int)(data['entry'][0]['resource']['birthDate'][0:4])
else:
    print("no age in this file")
    print(data['entry'][0]['resource']['id'])
'''
