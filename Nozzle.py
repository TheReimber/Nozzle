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
Exit_Ae = Throat_AreaOfThroat * np.sqrt((1 / Exit_Me ** 2) * (2 / (Engine_Y + 1) * (1 + ((Engine_Y - 1) / 2) * Exit_Me ** 2)) ** ((Engine_Y + 1) / (Engine_Y - 1)))
Exit_RE = np.sqrt(Exit_Ae / np.pi)
ExpansionRatio = Exit_Ae / Throat_AreaOfThroat
sqrtEpsilon = np.sqrt(Exit_RE / Throat_R)
Re = np.sqrt(Exit_Ae) * Throat_R
R1 = 1.5 * Throat_R
alpha = 15 * np.pi / 180
Xn = R1 * np.sin(alpha)
Yn = (Throat_R + R1) - (R1 * np.cos(alpha))
Ln = (Exit_RE - Throat_R + R1 * (np.cos(alpha) - 1)) / (np.tan(alpha))
L = Ln + R1 * np.sin(alpha)
L1 = R1 * np.sin(alpha)
XN = 0.0961429549062482
XE = 1.44886536002306

# General Varibles
XY_ListComp = []
NoOfPoints = 10
XnList = []
YnList = []
Xn2List = []
Yn2List = []
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



# Create an array List holding all the calculated points Based on the end Value of Xn and zero
# For Inlet and Throat
# Add Zero to first position
XnList.append(0)
for i in range(1,NoOfPoints, 1):
    XnList.append(Xn/NoOfPoints * i)

# Append with end calculated Value Xn
XnList.append(Xn)


# Create an array List holding all the calculated points Based on the end Value of Yn Taking into account the Zero value of Xn
# For Inlet and Throat
for j in range(0,NoOfPoints,1):
    YnList.append(R1 * (1 - np.cos(np.arcsin(XnList[j] / R1))) + Throat_R)

# Append with end calculated Value Yn
YnList.append(Yn)

# Create Second Part of Graph (Linear Section) for X (ignoring 0)

for k in range(0,NoOfPoints,1):

    if k == 0:
        Xn2List.append(Xn)
    elif k == 1:
        Xn2List.append(XN + (XE / (len(XnList)-2) * k))
    else:
        Xn2List.append(XN + ((XE-XN) / (len(XnList) - 2) * k))


for l in range(0,NoOfPoints,1):
    # Place Last item from previous non linear calc into firt item in new list

    if l == 0:
        Yn2List.append(YnList[len(YnList)-1])
    # Use previous item in new list for next list item calc
    if l >= 1:
        Yn2List.append(Yn2List[l-1]+(Exit_RE-Throat_R)/9)

# Add final item to list as it is a pre-calculated Item
Yn2List.append(Exit_RE)
# Combine Both lists into XnList and YnList
XnList.extend(Xn2List)
YnList.extend(Yn2List)

# Create DataFrame and load XnArray and YnArray in as X and Y values
snsList = []

for i in range(0,len(XnList)-1):
    snsList.append([XnList[i],YnList[i]])

dataframe = pd.DataFrame(snsList, columns=['X', 'Y'])
print(dataframe)
sns.lineplot(data=dataframe,x='X',y='Y')
plt.title("Linear Nozzle")
plt.show()

# Show Calculations on terminal
CalcVals()
