import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF

CalcList = []


# Define Functions
def CalcVals():
   # These are all calculations based on initial values of PO,To,Mol Mass,Cp,Pe,R_Uni and Throat_R
   # and placed in a list for printing to a pdf later

    CalcList.append(('Engine_PO_Pa'.ljust(20), Engine_PO_Pa))
    CalcList.append(('Engine_Pe_Pa'.ljust(20), Engine_Pe_Pa))
    CalcList.append(('Engine_Rspec'.ljust(20), Engine_Rspec))
    CalcList.append(('Engine_Y'.ljust(20), Engine_Y))
    CalcList.append(('Engine_rho_0'.ljust(20), Engine_rho_0))
    CalcList.append(('Engine_Thrust'.ljust(20), Engine_Thrust))
    CalcList.append(('Throat_M_Ex_Engine'.ljust(20), Throat_M_Ex_Engine))
    CalcList.append(('Throat_rho'.ljust(20), Throat_rho))
    CalcList.append(('Throat_sos'.ljust(20), Throat_sos))
    CalcList.append(('Exit_Te'.ljust(20), Exit_Te))
    CalcList.append(('Exit_Me'.ljust(20), Exit_Me))
    CalcList.append(('Exit_SpeedAtExit'.ljust(20), Exit_SpeedAtExit))
    CalcList.append(('Throat_M_dot'.ljust(20), Throat_M_dot))
    CalcList.append(('Exit_RE'.ljust(20), Exit_RE))
    CalcList.append(('Throat_AreaOfThroat'.ljust(20), Throat_AreaOfThroat))
    CalcList.append(('Throat_R'.ljust(20), Throat_R))
    CalcList.append(('Exit_Ae'.ljust(20), Exit_Ae))
    CalcList.append(('Exit_RE'.ljust(20), Exit_RE))
    CalcList.append(('ExpansionRation'.ljust(20), ExpansionRatio))
    CalcList.append(('sqrtEpsilon'.ljust(20), sqrtEpsilon))
    CalcList.append(('Re'.ljust(20), Re))
    CalcList.append(('R1'.ljust(20), R1))
    CalcList.append(('alpha'.ljust(20), alpha))
    CalcList.append(('Xn'.ljust(20), Xn))
    CalcList.append(('Yn'.ljust(20), Yn))
    CalcList.append(('Ln'.ljust(20), Ln))
    CalcList.append(('L'.ljust(20), L))
    CalcList.append(('L1'.ljust(20), L1))

def CombineList(list1,list2):
    list1.extend(list2)
    return list1
def plot(dataframe):
    sns.lineplot(data=dataframe,x='X',y='Y')
    plt.title("Linear Nozzle")
    # Save Plot as PDF
    plt.savefig('Plot.pdf')
    plt.show()
def CreateDataFrame(x,y):
    # Create DataFrame and load XnArray and YnArray in as X and Y values
    snsList = []
    for i in range(0,len(x)-1):
        snsList.append([x[i],y[i]])
    # Copy the snsList into a data frame

    dataframe = pd.DataFrame(snsList, columns=['X', 'Y'])
    dataframe.style.set_properties(**{'text-align': 'left'})

    return dataframe
def CreateCalcFrame(calcs):

    dataframe = pd.DataFrame(calcs, columns=['Name'.ljust(20), 'Value'.ljust(12)])

    return dataframe


def pdf_print(fileToPrint,filename):
    print("Printing....")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 13)
    pdf.multi_cell(h=5.0, align='J', w=0, txt=fileToPrint, border=0)
    pdf.output(filename, 'F')
    print('Printing Done...')

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

# General Variables
XY_ListComp = []
NoOfPoints = 10
XnList = []
YnList = []
Xn2List = []
Yn2List = []


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

# Create Second Part of Graph (Linear Section) for X (Moving Xn directly into list first)

for k in range(0,NoOfPoints,1):

    if k == 0:
        Xn2List.append(Xn)
    elif k == 1:
        Xn2List.append(XN + (XE / (len(XnList)-2) * k))
    else:
        Xn2List.append(XN + ((XE-XN) / (len(XnList) - 2) * k))


for l in range(0,NoOfPoints,1):
    # Place Last item from previous non-linear calc into first item in new list

    if l == 0:
        Yn2List.append(YnList[len(YnList)-1])
    # Use previous item in new list for next list item calc
    if l >= 1:
        Yn2List.append(Yn2List[l-1]+(Exit_RE-Throat_R)/9)

# Add final item to list as it is a pre-calculated Item
Yn2List.append(Exit_RE)

################
# Function Calls
################

# Combine XnList and Xn2List together then YnList and Yn2List
XnList = CombineList(XnList,Xn2List)
YnList =CombineList(YnList,Yn2List)
# Show Calculations on terminal
CalcVals()


#Get Dataframe Data
DataFrame = CreateDataFrame(XnList,YnList)
# Print DataFrame and CalcList to files within program root directory
pdf_print(str(DataFrame),'DataFrame')
pdf_print(str(CreateCalcFrame(CalcList)),'Calculations')

#Plot Data frame
plot(DataFrame)


