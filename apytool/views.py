from django.shortcuts import render, redirect #Httpresponse, Httpresponseredirect
#Importing the database and form render
from .models import *
from .forms import *
#plot rendering modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np
# Create your views here.
from scipy.optimize import curve_fit
#import sklearn 
#from sklearn import skel

form_data_var = None
ypr_data_var = None

def func_index(request):
	return render(request, 'apytool/index.html')

def func_shrimp_trend(request):
	if request.method == 'POST':
		form          = DetailForm(request.POST)
		global form_data_var
		if form.is_valid():
			form_data_var = {
			'species': form.cleaned_data.get('species'), 
			'State': form.cleaned_data.get('State')
			} #obtaining the state from the cleaned data
			form = DetailForm()
			return redirect('results_page')
	
	else:
		form    = DetailForm()
	
	context = {'form' : form}
	return render(request, 'apytool/shrimp_trends.html', context)

def result_plots(request):
	global form_data_var
	if form_data_var is None:
		return redirect('shrimp_trend_page')
	context = {'productivity_graph': return_productivity_graph(),
	'production_graph': return_production_graph(),
	'area_graph': return_area_graph(),
	'prod_growth_rate': compute_growth_rate('Production'), 
	'cv_analysis': coeff_of_var()}
	form_data_var = None 
	return render(request, 'apytool/results.html', context)

def return_productivity_graph():
	global form_data_var
	queryset_data = ShrimpData.objects.filter(
		species=form_data_var['species'],
		State=form_data_var['State']).order_by('Year').values()
	x = []
	y = []
	z = []
	#since queryset is not too large



	for i in range(len(queryset_data)):
		x.append(queryset_data[i]['Area'])
		y.append(queryset_data[i]['Production'])
		z.append(queryset_data[i]['Year'])

	x = np.array(x, dtype=np.float64)
	y = np.array(y, dtype=np.float64)
	r = y/x
	r[np.isnan(r)] = 0 
	z = np.array(z, dtype=np.int32)
	fig = plt.figure()
	
	plt.plot(z,r, "-o")
	plt.grid()
	#plt.plot(z,y)
	plt.xlabel('Year')
	plt.ylabel('Yield in Met. Tonnes/ha')
	plt.title('PRODUCTIVITY of ' + form_data_var['State']+ ' Over 2010 to 2021')

	opt,cov_mat = curve_fit(lin_yield_fit, z-2010, r)

	s           = opt[0]*(z-2010)+opt[1]
	plt.plot(z, s)

	plt.legend(['data', 'regression'])
	plt.gcf().autofmt_xdate()
	imgdata = StringIO()
	fig.savefig(imgdata, format='svg')
	imgdata.seek(0)

	data = imgdata.getvalue()
	return data

#Doing the job of curve fitting to the productivity data	
def lin_yield_fit(x, a, b):
	return a*x+b

def return_area_graph():
	global form_data_var
	queryset_data = ShrimpData.objects.filter(
		species=form_data_var['species'],
		State=form_data_var['State']).order_by('Year').values()
	x = []
	z = []
	#since queryset is not too large
	#for loop time doesn't add much overhead here 
	for i in range(len(queryset_data)):
		x.append(queryset_data[i]['Area'])
		z.append(queryset_data[i]['Year'])

	x = np.array(x, dtype=np.float64)
	z = np.array(z)
	fig = plt.figure()
	plt.plot(z,x, "-o")
	#plt.plot(z,y)
	plt.xlabel('Year')
	plt.ylabel('Area of cultivation in Hectares')
	plt.title('AREA UNDER CULTIVATION in '+form_data_var['State']+' Over 2010 to 2021')
	plt.gcf().autofmt_xdate()
	imgdata = StringIO()
	fig.savefig(imgdata, format='svg')
	imgdata.seek(0)

	data = imgdata.getvalue()
	return data

def coeff_of_var():
	global form_data_var
	queryset_data = ShrimpData.objects.filter(
		species=form_data_var['species'],
		State=form_data_var['State']).order_by('Year').values()
	x = []
	y = []
	z = []

	for i in range(len(queryset_data)):
		x.append(queryset_data[i]['Area'])
		y.append(queryset_data[i]['Production'])
		z.append(queryset_data[i]['Year'])

	x = np.array(x, dtype=np.float64)
	y = np.array(y, dtype=np.float64)
	r = y/x
	r[np.isnan(r)] = 0
	return(round(100*(np.std(x)/np.mean(x)),2), 
		round(100*(np.std(y)/np.mean(y)),2),
		round(100*(np.std(r)/np.mean(r)),2), round(np.corrcoef(x,y)[1,0],2), round(100*(x[-1]-x[0])/x[0],2))

def return_production_graph():
	global form_data_var
	queryset_data = ShrimpData.objects.filter(
		species=form_data_var['species'],
		State=form_data_var['State']).order_by('Year').values()
	y = []
	z = []
	#since queryset is not too large
	#for loop time doesn't add much overhead here 
	for i in range(len(queryset_data)):
		y.append(queryset_data[i]['Production'])
		z.append(queryset_data[i]['Year'])

	y = np.array(y, dtype=np.float64)
	z = np.array(z)
	fig = plt.figure()
	
	plt.plot(z,y, "-o")
	#plt.plot(z,y)
	plt.xlabel('Year')
	plt.ylabel('Production in Met. Tonnes')
	plt.title('PRODUCTION in '+form_data_var['State']+' Over 2010 to 2021')
	plt.gcf().autofmt_xdate()
	imgdata = StringIO()
	fig.savefig(imgdata, format='svg')
	imgdata.seek(0)

	data = imgdata.getvalue()
	return data

def compute_growth_rate(keyword):
	global form_data_var
	queryset_data = ShrimpData.objects.filter(
		species=form_data_var['species'],
		State=form_data_var['State']).order_by('Year').values()
	k             = len(queryset_data)
	if queryset_data[0][keyword] ==0:
		queryset_data[0][keyword] = 1
	return '{:.04f}'.format(100*((queryset_data[k-1][keyword]/queryset_data[0][keyword])**(1/(int(queryset_data[k-1]['Year'])-int(queryset_data[0]['Year'])))-1))




#YIELDS AND PARAMETER ANALYSIS 

def func_shrimp_seaweed(request):
	'''Render the shrimp seaweed selection page'''
	return render(request, 'apytool/shrimp_seaweed_sel.html')

def YPR_shrimp_details(request):
	''' TAKE THE SHRIMP DATA IN THE FORM AND PROCESS IT '''
	global ypr_data_var
	if request.method == "POST":
		ypr_data_var = request.POST
		ypr_data_var = dict(ypr_data_var)
		ypr_data_var.update({'organism': 'Shrimps'})
		return redirect("ypr_results_page")
	return render(request, 'apytool/param_shrimp.html')

def YPR_seaweed_details(request):
	''' TAKE THE SEAWEED DATA IN THE FORM AND PROCESS IT '''
	global ypr_data_var
	if request.method == "POST":
		ypr_data_var = request.POST
		ypr_data_var = dict(ypr_data_var)
		ypr_data_var.update({'organism': 'Seaweeds'})
		return redirect("ypr_results_page")
	return render(request, 'apytool/param_seaweed.html')

def func_YPR_results(request):
	''' THE COMPUTATION OF THE ESTIMATED BIOMASS AND GROWTH RATES ARE MADE
	BY CALLING THE HELPER FUNCTIONS WHICH CALCULATE THE RESULTS'''
	global ypr_data_var
	if ypr_data_var is None:
		return redirect('shrimps_seaweeds_sel_page')
	else:
		if ypr_data_var['organism'] == "Shrimps":
			context = {
			'species': ypr_data_var['species'][0],
			'temperature': ypr_data_var['temperature'][0],
			'stock_den': ypr_data_var['stock_den'][0],
			'ph': ypr_data_var['ph'][0],
			'salinity': ypr_data_var['salinity'][0],
			'fqn': ypr_data_var['fqn'][0],
			'fq': ypr_data_var['fq'][0],
			'doc': ypr_data_var['doc'][0],
			'organism': ypr_data_var['organism']
			}
			(gr, final_weight, surv_shrimps, biomass, fcr) = ypr_calc(temperature = context['temperature'], 
			stock_den = context['stock_den'], ph = context['ph'], salinity = context['salinity'], fqn = context['fqn'], fq = context['fq'], doc = context['doc'],
			intensity = None, depth = None, secchi = None, organism = context['organism'])
			context['final_weight'] = round(final_weight,2)
			context['surv_shrimps'] = round(surv_shrimps,2)
			context['biomass']      = round(biomass,2)
			context['fcr']          = round(fcr,2)
			context['gr']           = round(gr,2)
			ypr_data_var = None
			return render(request, "apytool/ypr_results_page.html", context)
		else:
			context = {
			'species': ypr_data_var['species'][0],
			'temperature': ypr_data_var['temperature'][0],
			'stock_den': ypr_data_var['stock_den'][0],
			'intensity': ypr_data_var['intensity'][0],
			'secchi': ypr_data_var['secchi'][0],
			'depth': ypr_data_var['depth'][0],
			'doc': ypr_data_var['doc'][0],
			'organism': ypr_data_var['organism']
			}
			(growth_rate, biomass) = ypr_calc(temperature = context['temperature'], 
			stock_den = context['stock_den'], ph = None, salinity = None, fqn = None, fq = None, doc = context['doc'],
			intensity = context['intensity'], depth = context['depth'], secchi = context['secchi'], organism = context['organism'])
			context['growth_rate'] = round(growth_rate,2)
			context['biomass']      = round(biomass,2)
			ypr_data_var = None
			return render(request, "apytool/ypr_results_page.html", context)






#temp in the below functions primarily points to the temperory variables in python
#ALL THE BELOW FROM HERE ARE SHRIMP MODELS



def ypr_calc(temperature, stock_den, ph, salinity, fqn, fq, doc, intensity, depth, secchi, organism):
	if organism == "Shrimps":
		gr = shrimp_temp_model(temperature)*shrimp_stden_model(stock_den)*shrimp_sal_model(salinity)
		max_gr_ph = 0
		for i in range(0, 1400):
			max_gr_ph = max(max_gr_ph, shrimp_ph_model(i/100))
		norm_ph = shrimp_ph_model(ph)/max_gr_ph
		gr = gr*norm_ph
		gr = gr*float(fq)
		final_weight = gr*(float(doc)/7)
		surv_shrimps = float(stock_den)*shrimp_surv_stden(stock_den)
		biomass = float(surv_shrimps)*float(final_weight)
		fcr = float(fqn)*1000/biomass
		return (gr, final_weight, surv_shrimps, biomass, fcr)
	else:
		temperature = float(temperature)
		stock_den   = float(stock_den)
		doc         = float(doc)
		intensity   = float(intensity)
		depth       = float(depth)
		turbidity   = float(secchi)
		gr = seaweed_temp_model(temperature)
		gr *= seaweed_turb_model(secchi) #normalising with respect to the secchi depth
		x = np.linspace(10,300,4000) 
		max_intensity = 0
		for i in x:
			max_intensity = max(max_intensity, seaweed_light_model(i))
		gr *= seaweed_light_model(intensity)/max_intensity #normalising wrt to the light intensity
		x = np.linspace(0.01,5,4000) 
		max_depth = 0
		for i in x:
			max_depth = max(max_depth, seaweed_depth_model(i))
		gr *= seaweed_depth_model(depth)/max_depth
		biomass = stock_den * (1+(gr/100))**doc
		return (gr, biomass)




def shrimp_temp_model(temp):
	temp = float(temp)
	x_small = [23, 27, 30]
	y_small = [0.8, 1.78, 2.1]
	z_small = np.polyfit(x_small,y_small,2)
	grs_22 = z_small[0]*(22)**2+z_small[1]*(22)+z_small[2]
	grs_35 = z_small[0]*(35)**2+z_small[1]*(35)+z_small[2]
	if temp>=22 and temp<=35:
		grs = z_small[0]*temp**2+z_small[1]*temp+z_small[2]
	elif temp<22:
		grs = grs_22*np.exp((temp-22)) #based on intuitive understanding
	else:
		grs = grs_35*np.exp(-(temp-35)) #lower rate of decay because even the L.vannamei species can tolerate 37 degree C, so more chances of survival on this range
	x_med = [23, 27, 30]
	y_med = [0.42, 1.1, 1.25]
	z_med = np.polyfit(x_med,y_med,2)
	grm_22 = z_med[0]*(22)**2+z_med[1]*(22)+z_med[2]
	grm_35 = z_med[0]*(35)**2+z_med[1]*(35)+z_med[2]
	if temp>=22 and temp<=35:
		grm = z_med[0]*temp**2+z_med[1]*temp+z_med[2]
	elif temp<22:
		grm = grm_22*np.exp((temp-22)) #based on intuitive understanding
	else:
		grm = grm_35*np.exp(-(temp-35))
	x_large = [23, 27, 30]
	y_large = [0.37, 0.62, 0.5]
	z_large = np.polyfit(x_large,y_large,2)
	grl_22 = z_large[0]*(22)**2+z_large[1]*(22)+z_large[2]
	grl_35 = z_large[0]*(35)**2+z_large[1]*(35)+z_large[2]
	if temp>=22 and temp<=35:
		grl = z_large[0]*temp**2+z_large[1]*temp+z_large[2]
	elif temp<22:
		grl = grl_22*np.exp((temp-22)) #based on intuitive understanding
	else:
		grl = grl_35*np.exp(-(temp-35))
	return 0.2*grs+0.35*grm+0.45*grl

def shrimp_stden_model(temp):
	temp = float(temp)
	x_stock = [30, 40, 50, 60, 70, 80]
	y_stock = [1.5248, 1.394, 1.167, 1.0675, 0.9905, 0.856]
	z_stock = np.polyfit(x_stock, y_stock, 2)
	max_gr_temp = 0
	#normalising with the temperature data growth rate 
	for i in range(500,5000):
		max_gr_temp = max(max_gr_temp, shrimp_temp_model(i/100))
	z_stock = np.array(z_stock)/max_gr_temp # normalised with respect to temperature
	return z_stock[0]*temp**2+z_stock[1]*temp+z_stock[2] 

def shrimp_surv_stden(temp):
	temp = float(temp)
	x_surv = [30,40,50,60,70,80]
	y_surv = [0.95,0.925,0.91,0.9,0.85,0.82]
	z_surv = np.polyfit(x_surv, y_surv, 2)
	return z_surv[0]*temp**2+z_surv[1]*temp+z_surv[2]

def shrimp_sal_model(temp):
	temp = float(temp)
	x_sal = [5, 15, 25, 35, 49]
	y_sal = [2.146, 2.12, 1.834, 1.864, 1.632]
	z_sal = np.polyfit(x_sal, y_sal, 1)
	return (z_sal[0]*temp+z_sal[1])/np.max(y_sal)

def shrimp_ph_model(temp):
	temp = float(temp)
	x_ph = [5.1, 5.9, 6.5, 7, 8]
	y_ph = [0.28, 0.5, 0.52, 0.55, 0.67]
	z_ph = np.polyfit(x_ph, y_ph, 2)
	gr_6 = z_ph[0]*(6.5)**2+z_ph[1]*(6.5)+z_ph[2]
	gr_9 = z_ph[0]*(9.5)**2+z_ph[1]*(9.5)+z_ph[2]
	if temp>=6.5 and temp<=9.5:
		return z_ph[0]*temp**2+z_ph[1]*temp+z_ph[2]
	elif temp<6.5:
		return gr_6*np.exp(2*(temp-6.5))
	else:
		return gr_9*np.exp(-2*(temp-9.5))






#ALL THE BELOW ARE SEAWEED MODELS

def seaweed_depth_model(temp):
	temp = float(temp)
	x = [0.2, 1, 2, 3, 4, 5]
	y = [5.8, 4, 3.5, 3, 2.7, 2.4]
	z = np.polyfit(x,y,2)
	return z[0]*temp**2+z[1]*temp+z[2]

def seaweed_turb_model(temp):
	temp = float(temp)
	turb = secchi_vs_turb(temp)/secchi_vs_turb(0.1) 
	# 0.1 is the secchi depth having highest turbidity and 
	#we are normalising it with that 
	x_turb = [1, 0.5, 0.25]
	y_turb = [0.329, 0.252, 0.181]
	z_turb = np.polyfit(x_turb,y_turb,2)
	#irradiance = 1-turbidity
	irradiance = 1-turb
	z_max = z_turb[0]+z_turb[1]+z_turb[2]
	return (z_turb[0]*irradiance**2+z_turb[1]*irradiance+z_turb[2] )/z_max


def secchi_vs_turb(x):
	'''converts secchi depth to turbidity'''
	return (x/3.39)**(-1/0.637)

def seaweed_light_model(temp):
	temp=float(temp)
	x_light = [113, 84.786, 56.524, 28.262]
	y_light = [0.363, 0.318, 0.276, 0.166]
	z_light = np.polyfit(x_light,y_light,2)
	return z_light[0]*temp**2+z_light[1]*temp+z_light[2]

def seaweed_temp_model(temp):
	temp = float(temp)
	x = [20, 22.5, 24.5, 25, 26.5, 27.5]
	y = [3, 5, 7, 6, 6, 5.5]
	z = np.polyfit(x,y,2)
	y_20 = z[0]*20**2+z[1]*20+z[2]
	y_30 = z[0]*30**2+z[1]*30+z[2]
	if temp <= 20:
		return y_20*np.exp(0.5*(x-20))
	elif temp>=30:
		return y_30*np.exp(-0.5*(x-30))
	else:
		return z[0]*temp**2+z[1]*temp+z[2]


