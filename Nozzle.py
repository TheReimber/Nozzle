import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
# Set Constant Variables
Engine_PO = 15
Engine_TO = 2500
Engine_MolMass = 27
Engine_Cp = 2112
Engine_Pe = 0.372
Engine_R_uni = 8314
Throat_M = 1

# Setup Variables Calculated values
Engine_Thrust = 4.5 * pow(10, 5)
Engine_PO_Pa = 15 * 101325
Engine_Pe_Pa = 0.372 * 101325
Engine_Rspec = Engine_R_uni / Engine_MolMass
Engine_Y = Engine_Cp / (Engine_Cp - Engine_Rspec)
Engine_rho_0 = Engine_PO_Pa / (Engine_Rspec * Engine_TO)
Throat_M_Ex_Engine = np.sqrt((2 / (Engine_Y - 1)) * (pow((Engine_PO / Engine_Pe), (Engine_Y - 1) / Engine_Y) - 1))
Throat_rho = Engine_rho_0 * (pow((2 / (Engine_Y + 1)), (1 / (Engine_Y - 1))))
Throat_sos = np.sqrt(Engine_Y * Engine_Rspec * Engine_TO * (2 / (Engine_Y + 1)))
Exit_Me = np.sqrt(2 * (((Engine_PO / Engine_Pe) ** ((Engine_Y - 1) / Engine_Y) - 1) / (Engine_Y - 1)))
Exit_Te = Engine_TO * pow((Engine_Pe / Engine_PO), ((Engine_Y - 1) / Engine_Y))
Exit_SpeedAtExit = Throat_M_Ex_Engine * np.sqrt((Engine_Y * Engine_Rspec * Exit_Te))
Throat_M_dot = Engine_Thrust / Exit_SpeedAtExit
Throat_AreaOfThroat = Throat_M_dot / (Throat_sos * Throat_rho)
Throat_R = np.sqrt(Throat_AreaOfThroat / np.pi)
Exit_Ae = Throat_AreaOfThroat * np.sqrt(
    (1 / Exit_Me ** 2) * (2 / (Engine_Y + 1) * (1 + ((Engine_Y - 1) / 2) * Exit_Me ** 2)) ** (
                (Engine_Y + 1) / (Engine_Y - 1)))
Exit_RE = np.sqrt(Exit_Ae / np.pi)
ExpansionRatio = Exit_Ae / Throat_AreaOfThroat
sqrtEpsilon = np.sqrt(Exit_RE / Throat_R)
Re = np.sqrt(Exit_Ae) * Throat_R
R1 = 1.5 * Throat_R
alpha = 15 * np.pi / 180
Xn = R1 * np.sin(alpha)
Yn = (Throat_R + R1) - (R1 * np.cos(alpha))
Ln = (Exit_RE - Throat_R + R1 * (np.cos(alpha) - 1)) / (np.tan(alpha))
L = ExpansionRatio + R1 * np.sin(alpha)
L1 = R1 * np.sin(alpha)

def CalcVals():
    print('\nThese are all calculations based on initial values of PO,To,Mol Mass,Cp,Pe,R_Uni and Throat_R :-\n')
    print('Engine_PO_Pa', Engine_PO_Pa)
    print('Engine_Pe_Pa', Engine_Pe_Pa)
    print('Engine_Rspec', Engine_Rspec)
    print('Engine_Y', Engine_Y)
    print('Engine_rho_0', Engine_rho_0)
    print('Engine_Thrust', Engine_Thrust)
    print('Throat_M_Ex_Engine ', Throat_M_Ex_Engine)
    print('Throat_rho', Throat_rho)
    print('Throat_sos', Throat_sos)
    print('Exit_Te', Exit_Te)
    print('Exit_Me', Exit_Me)
    print('Exit_SpeedAtExit', Exit_SpeedAtExit)
    print('Throat_M_dot', Throat_M_dot)
    print('Exit_RE', Exit_RE)
    print('Throat_AreaOfThroat', Throat_AreaOfThroat)
    print('Throat_R', Throat_R)
    print('Exit_Ae', Exit_Ae)
    print('Exit_RE', Exit_RE)
    print('ExpansionRation', ExpansionRatio)
    print('sqrtEpsilon', sqrtEpsilon)
    print('Re', Re)
    print('R1', R1)
    print('alpha', alpha)
    print('Xn', Xn)
    print('Yn', Yn)
    print('Ln', Ln)
    print('L', L)
    print('L1', L1)
    print(XnArray)
    print(YnArray)






NoOfPoints = 80
XnList = []
YnList = []

# Create an array holding all the calculated points Based on the end Value of Xn and zero
for i in range(0, NoOfPoints -1, 1):
    XnList.append(Xn / NoOfPoints * i)
XnArray = np.array([XnList])

# Create an array holding all the calculated points Based on the end Value of Yn Taking into account the Zero value of Xn
for j in range(0,NoOfPoints-1,1):
    if j >= 1:
        YnList.append(R1*(1-np.cos(np.arcsin(XnList[j-1]/R1)))+Throat_R)
    else:
        YnList.append(R1 * (1 - np.cos(np.arcsin(0 / R1))) + Throat_R)
YnArray = np.array([YnList])

# Create DataFrame and load XnArray and YnArray in as X and Y values
snsList = []

for i in range(0,len(XnList)):
    snsList.append([XnList[i],YnList[i]])

dataframe = pd.DataFrame(snsList, columns=['X', 'Y'])
print(dataframe)

sns.lineplot(data=dataframe,x='X',y='Y')
sns.pointplot()
plt.show()




# CalcVals()
