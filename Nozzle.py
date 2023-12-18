import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set Constant Variables
Engine_PO = 15
Engine_TO = 2500
Engine_MolMass = 27
Engine_Cp = 2112
Engine_Pe = 0.372
Engine_R_uni = 8314

Throat_M = 1
# Setup Variables Calculated values
Engine_Thrust = 4.5 * pow(10,5)
Engine_PO_Pa = 15 * 101325
Engine_Pe_Pa = 0.372 * 101325
Engine_Rspec = Engine_R_uni / Engine_MolMass
Engine_Y = Engine_Cp / (Engine_Cp - Engine_Rspec)
Engine_rho_0 = Engine_PO_Pa / (Engine_Rspec * Engine_TO)

Throat_M_Ex_Engine = np.sqrt((2/(Engine_Y-1)) * (pow((Engine_PO / Engine_Pe), (Engine_Y - 1) / Engine_Y) - 1))
Throat_M_dot =0
####    Throat_rho = Engine_rho_0 * (pow((2/(Engine_Y +1)),(1/(Engine_Y-1)))) #######
Throat_sos = np.sqrt(Engine_Y * Engine_Rspec * Engine_TO * (2/(Engine_Y+1)))
Throat_AreaOfThroat = 0
Throat_R =0

Exit_Ae = 0
Exit_Me = 0
Exit_RE = 0
Exit_SpeedAtExit = 0
Exit_Te = Engine_TO * pow((Engine_Pe/Engine_TO),(Engine_Y-1)/Engine_Y)
# print('Engine_PO_Pa',Engine_PO_Pa)
# print('Engine_Pe_Pa',Engine_Pe_Pa)
# print('Engine_Rspec',Engine_Rspec)
# print('Engine_Y',Engine_Y)
# print('Engine_rho_0',Engine_rho_0)
# print('Engine_Thrust',Engine_Thrust)
# print('Throat_M_Ex_Engine ',Throat_M_Ex_Engine )
# print('Throat_rho',Throat_rho)
# print('Throat_sos',Throat_sos)
print('Exit_Te',Exit_Te)
