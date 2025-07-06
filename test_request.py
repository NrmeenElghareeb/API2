import requests
url = "http://127.0.0.1:5000/predict"


#url = "http://127.0.0.1:500-00/predict"

data = {
 
    "pred_x_km": 12345.6,
    "pred_y_km": 23456.7,
    "pred_z_km": 34567.8,
    "pred_vx_km_s": 1.1,
    "pred_vy_km_s": 2.2,
    "pred_vz_km_s": 3.3,
    "KP_SUM_mean": 4.0,
    "AP_AVG_mean": 5.0,
    "F10.7_ADJ_mean": 6.0,
    "F10.7_ADJ_LAST81_mean":7.0,
    "F10.7_OBS_mean": 8.0,
    "grav_potential":-9999.0,
    "grav_x": 0.1,
    "grav_y":0.2,
    "grav_z":0.3,
    "grav_magnitude": 9.8,
   
}


response = requests.post(url, json=data)

print("Response:")
print(response.json())






'''

data = {
{
  "potential": 58356411.778024115,
  "gravX": 0.00004837369872153629,
  "gravY": -0.000029448246250363928,
  "gravZ": 8.545039563878197,
  "gravMagnitude": 8.545039564065862,
  "x": -1613.9558559286436,
  "y": -5837.499054917121,
  "z": 3160.6908360275024,
  "vx": -1.998970385715914,
  "vy": -3.095291680183207,
  "vz": -6.69099046781726,
  "bb": 0.00052866
}


{
  "kP_SUM_to_use": 191.83333333333334,
  "aP_AVG": 10.5,
  "f107_ADJ": 130.55555555555554,
  "f107_ADJ_LAST81": 138.14999999999998,
  "f107_OBS": 126.34444444444443
}



 #"b_star":0.000018869
















    
    "pred_x_km": 1000,
    "pred_y_km": 2000,
    "pred_z_km": 3000,
    "pred_vx_km_s": 1.5,
    "pred_vy_km_s": 2.5,
    "pred_vz_km_s": -1.0,
    "KP_SUM_mean": 2.3,
    "AP_AVG_mean": 4.5,
    "F10.7_ADJ_mean": 150.0,
    "F10.7_ADJ_LAST81_mean": 145.0,
    "F10.7_OBS_mean": 148.0,
    "ballistic_coefficient": 0.01,
    "grav_potential": -7200000,
    "grav_x": 0.1,
    "grav_y": 0.2,
    "grav_z": 0.3,
    "grav_magnitude": 9.81
}
'''