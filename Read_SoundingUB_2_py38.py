##programa per llegir els arxius txt del sondeig de la facultat de física UBA
##genera els gàfics de temperatura vers la pressió

##By Albert Garcia Benadí


import os
import glob

from matplotlib import gridspec
##import matplotlib.pyplot as plt
##import numpy as np
import datetime
from datetime import datetime,date
import gmplot
import matplotlib.pyplot as plt
from metpy.plots import Hodograph,SkewT
from metpy.units import units
import numpy as np
import metpy.calc as mpcalc
from scipy.signal import medfilt
from metpy import constants as mpconsts


from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# create new figure, axes instances.




def pressure_to_height_std(pressure):
    t0 = 288. * units.kelvin
    p0 = 1013.25 * units.hPa
    r"""Convert pressure data to heights using the U.S. standard atmosphere [NOAA1976]_.

    The implementation uses the formula outlined in [Hobbs1977]_ pg.60-61.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure

    Returns
    -------
    `pint.Quantity`
        The corresponding height value(s)

    Notes
    -----
    .. math:: Z = \frac{T_0}{\Gamma}[1-\frac{p}{p_0}^\frac{R\Gamma}{g}]

    """
    gamma = 6.5 * units('K/km')
    return (t0 / gamma) * (1 - (pressure / p0).to('dimensionless')**(
        mpconsts.Rd * gamma / mpconsts.g))




def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])




def ReadFile(fileid):

    
    fid5=fileid

    te1=fid5[-12:-6]
    te2=fid5[-6:-4]
    T1=te1+te2
    
    temps1=datetime.strptime(T1,'%y%m%d%H')


    f=open(fid5,'r')



    line=f.readline()
    LIST=line.split('\t')
    H=[]#ALTURA
    P=[]#PRESSIO(mmbar o hPa)
    T=[]#temperatura (Celsius)
    HUM=[]#Humedad %
    Td=[]#temp de rosada
    VelM=[]#modulo del viento en nudos
    VelD=[]#dirección del viento en grados
    lat=[]
    lon=[]


    while True:
    

        line=f.readline()
##        print('LINE',line)
        if not line:
            break
        VAR=line.split('\t')
##        print('VAR',VAR)
        H.append(float(VAR[1]))
        P.append(float(VAR[2]))
        T.append(float(VAR[3]))
        HUM.append(float(VAR[4]))
        Td.append(float(VAR[5]))
        VelM.append(float(VAR[6]))#Probo si son nusos
        VelD.append(float(VAR[7]))
        lat.append(float(VAR[10]))
        lon.append(float(VAR[11]))
##        print('pressure',P)
    
    f.close()

    H=np.array(H)
    P=np.array(P)
    T=np.array(T)
    Td=np.array(Td)
    VelM=np.array(VelM)
    VelD=np.array(VelD)
    lat=np.array(lat)
    lon=np.array(lon)

    
##    print(p_m)
##    print(lat)
##    fig=plt.figure()
##    ax4=fig.add_axes([0.1,0.1,0.8,0.8])
##    # setup mercator map projection.
##    m = Basemap(llcrnrlon=-6.,llcrnrlat=35.,urcrnrlon=6.,urcrnrlat=45.,\
##                rsphere=(6378137.00,6356752.3142),\
##                resolution='i',projection='merc',\
##                lat_0=41.,lon_0=2.)
##    x, y = m(lon,lat)
##    m.scatter(x,y,2,marker='o',color='r')
####    for j in range(len(x)):
####        m.plot(x[j],y[j],linewidth=0.8,color='r')
##
##    m.drawcoastlines()
##    m.fillcontinents(lake_color='#99ffff',alpha=0.4)
##    # draw parallels
##    m.drawparallels(np.arange(35,45,2),labels=[1,0,0,0])
##    # draw meridians
##    m.drawmeridians(np.arange(-6,6,2),labels=[0,0,0,1])
##
##    plt.show()


########    DELETE THE POINTS WHERE THE DOWNWARDS SGLOBUS MOVEMENTS


    test=[]
    for k in range(len(P)-1):
        test.append(P[k]-P[k+1])
    test=np.asarray(test)
##    print(test)
    if any(test<0):
##        print('\nhi haura problema\n')
        Pc=[];Tc=[];Tdc=[];VelMc=[];VelDc=[];Hc=[]
        for i in range(len(P)):
            if i==0:
                Pc.append(P[i]);Tc.append(T[i]);Tdc.append(Td[i]);VelMc.append(VelM[i]);VelDc.append(VelD[i]);Hc.append(H[i])
            else:
                if P[i]-Pc[-1]<=0:
                    Pc.append(P[i]);Tc.append(T[i]);Tdc.append(Td[i]);VelMc.append(VelM[i]);VelDc.append(VelD[i]);Hc.append(H[i])
        P=np.copy(Pc)
        T=np.copy(Tc)
        Td=np.copy(Tdc)
        VelM=np.copy(VelMc)
        VelD=np.copy(VelDc)
        H=np.copy(Hc)
        

##        for i in range(len(test)):
##            if test[i]<0:
##                print('index xungo',i,'altura',H[i])
##                Index=i-2
##                break
##        p_m = P[0:Index] * units.hPa
##        t_m= T[0:Index] * units.degC
##        td_m=Td[0:Index] * units.degC


        
    
    p_m = P * units.hPa
    t_m= T * units.degC
    td_m=Td * units.degC

        
    U=-1.*np.sin(VelD*np.pi/180)*VelM
    V=-1.*np.cos(VelD*np.pi/180)*VelM
    u, v = mpcalc.wind_components(VelM*units('m/s'), VelD*units.deg)#the result i u and v in knots. it is necessary introduce velD in radians
    ##u, v = mpcalc.wind_components(VelM*units.kt, VelD*units.deg)#the result i u and v in knots. it is necessary introduce velD in radians
    prof = mpcalc.parcel_profile(p_m, t_m[0], td_m[0])
    
    
##    print('Prof',prof,np.shape(prof))
    minPres=np.min(P)
    if minPres<100.:
        MinPres=100.
    else:
        MinPres=minPres
    
    IndxMin=np.where(P<=MinPres)[0];indxMin=IndxMin[0]
##    print('\n Minima pressio',MinPres,'al index ',indxMin,'\n')
    In850=np.where(P-850.<0)[0];ind850=In850[0]-1
##    if isnull(np.where(P-500.<0)[0]):
##        ind500=np.nan
##    else:
    In500=np.where(P-500.<0)[0];ind500=In500[0]-1
####    In300=np.where(P-300.<0)[0];ind300=In300[0]-1
    In600=np.where(P-600.<0)[0];ind600=In600[0]-1
##    In750=np.where(P-750.<0)[0];ind750=In750[0]-1
    In700=np.where(P-700.<0)[0];ind700=In700[0]-1
##    In650=np.where(P-650.<0)[0];ind650=In650[0]-1
##    In645=np.where(P-645.<0)[0];ind645=In645[0]-1
    
    if P[0]<1000.:#la pressio a H[0] es inferior a 1000 hPa
        t1=T[0];t2=T[1];p1=P[0];p2=P[1];
        t1000=t1+((1000-p1)*(t2-t1) /(p2-p1))
        td1=Td[0];td2=Td[1]
        td1000=td1+((1000-p1)*(td2-td1) /(p2-p1))
        ind1000=0
    else:
        
        In1000=np.where(P-1000.<0)[0];ind1000=In1000[0]-1
        t1=T[ind1000];t2=T[ind1000+1];p1=P[ind1000];p2=P[ind1000+1];
        t1000=t1+((1000-p1)*(t2-t1) /(p2-p1))
        td1=Td[ind1000];td2=Td[ind1000+1]
        td1000=td1+((1000-p1)*(td2-td1) /(p2-p1))
        

##    print(t1000,td1000,P[0])
##    print('temperatura a 850 hPa',T[ind850])
    fig = plt.figure(figsize=(16., 8.0), dpi=250)
##    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=5,ncols=4, wspace=0.3)
##    ax1 = fig.add_subplot(gs[0:2, 0:2])
    skew = SkewT(fig,rotation=45,subplot=gs[0:5, 0:2])

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(P, T, 'r',label='Ambient')
    skew.plot(P, Td, 'b',linestyle='dashed',label='Punt de rosada')
    skew.plot(P, prof, 'k',label='Partícula')  # Plot parcel profile

    lcl_pressure, lcl_temperature = mpcalc.lcl(p_m[0], t_m[0], td_m[0])
##    print(lcl_pressure, lcl_temperature)
######CALCUL DE LA NCC
    rat=mpcalc.saturation_mixing_ratio(1000*units.hPa,t1000*units.degC)
    rat1=mpcalc.saturation_mixing_ratio(p_m,td1000*units.degC)
    Inrat=np.where(rat1-rat<=0)[0]
##    print(Inrat)
    if len(Inrat)==0:
##        print('CCL','----','----')
        Prat=np.nan
                
    else:
        
##        print('CCL',P[Inrat[0]],Inrat[0])
        Prat=P[Inrat[0]]
    
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black',label='LCL')
    skew.ax.set_ylabel('Pressi'+u'\xf3'+' (hPa)')
    skew.ax.set_xlabel('Temperatura (Celsius)')

    
    # An example of a slanted line at constant T -- in this case the 0 isotherm
    l = skew.ax.axvline(0, color='k', linestyle='--', linewidth=2)
    
    # Add the relevant special lines to plot throughout the figure
    skew.plot_dry_adiabats(t0=np.arange(233, 533, 10) * units.K,alpha=0.25, color='orangered')
    skew.plot_moist_adiabats(t0=np.arange(233, 400, 5) * units.K,alpha=0.25, color='tab:green')
    skew.plot_mixing_lines(linestyle='dotted', color='tab:blue')
##    skew.ax.legend(loc='best')
    skew.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),fancybox=True, shadow=True, ncol=4)

##########CALCULS
    lfc_pressure,lfc_temperature=mpcalc.lfc(p_m, t_m, td_m,prof)

##    print('LCL ',lcl_pressure, lcl_temperature )        

    prcape, prcin = mpcalc.cape_cin(p_m, t_m, td_m, prof)
##    print('nou cape',prcape,prcin)

    pw = mpcalc.precipitable_water(p_m[:indxMin],td_m[:indxMin])
    if np.isnan(ind500):
        TT=np.nan
        Pam_K=np.nan
    else:
        TT=(T[ind850]-T[ind500])+(Td[ind850]-T[ind500])
        Pam_K=(T[ind850]-T[ind500])+Td[ind850]-(T[ind700]-Td[ind700])

    EL=mpcalc.el(p_m,t_m,td_m,prof)
##    print('tem,p per LI',T[ind500]*units.degC,prof[ind500].to('degC'))
    LI=T[ind500]*units.degC-prof[ind500].to('degC')
    li=float(LI/units.delta_degC)
##    print('LI',li)

##    print('prof',prof.to('degC'))

##    print('EL ', EL)

######    SUPERFICIE
    PrefISA=1013.25#hPa
    Pref=1000.#hPa segons meto.cat
##    print('temp i dp',t1000,td1000)
    tempot=mpcalc.equivalent_potential_temperature(Pref*units.hPa, (t1000+273.16)*units.K, (td1000+273.16)*units.K)
##    print('temperatura potencial a la superficie',tempot)


##############    ZONA INESTABLE
    mu=mpcalc.most_unstable_parcel(p_m, t_m, td_m)
##    print('parcela inestable',mu)
    tempot2=mpcalc.equivalent_potential_temperature(mu[0],mu[1].to(units.K),mu[2].to(units.K))

    capeU,cinU=mpcalc.most_unstable_cape_cin(p_m,t_m,td_m)
    if np.isnan(cinU):
        U_CIN=np.nan
    else:
        U_CIN=int(cinU/(units.joule / units.kilogram))
    if np.isnan(capeU):
        U_CAPE=np.nan
    else:
        U_CAPE=int(capeU/(units.joule / units.kilogram))
        
        
    #U_CAPE=int(capeU/(units.joule / units.kilogram))
    
    #print('u_cape',U_CAPE)


    
##################HODOGRAF
    h1_srh=3#en km
    indH=np.where(H-(h1_srh*1000.)>=0)[0];IndH=indH[0]
    
    uT=u[:IndH];vT=v[:IndH];VelMT=VelM[:IndH];PT=p_m[:IndH];HT=H[:IndH]*units.meter
##    print('\n igual que abans',indxMin,'\n')
    uTg=u[:indxMin];vTg=v[:indxMin];PTg=p_m[:indxMin]
##    print('\nhand e ser iguals els tres ',len(uTg),len(vTg),len(PTg),'\n')
    skew.plot_barbs(PTg[::100], uTg[::100], vTg[::100],xloc=1.0)#, y_clip_radius=0.03)
    skew.ax.set_ylim(np.max(p_m), MinPres)
    n_p=np.arange(1000,0,-100)
    n_h=pressure_to_height_std(n_p*units.hPa)

    
    for pp,hh in zip(n_p,n_h):
        skew.ax.text(-.17,pp,round(hh.m,1),transform=skew.ax.get_yaxis_transform(which='tick2'), color='b',fontsize=10,fontstyle='italic',va='center')
    skew.ax.text(-95,350,'Altitud (km)',fontsize=10,color='b',fontstyle='italic',rotation=90)
##    ax8=skew.ax.twinx()
##    ax8.spines['right'].set_position('center')
##    adjust_spines(skew.ax, ['right'])
    

    
##    ax8.set_frame_on(True)
##    ax8.patch.set_visible(False)
##    inv_h=H[::-1]
##    print('altura',inv_h)
##    new_h=p2h(p_m.ito('mb'))
##    ax8.plot(new_h, T[0:Index], 'r',label='Temperatura')
##    ax8.set_ylim(np.min(new_h),np.max(new_h))

##    rangeH=np.arange(0,15,1)*units.km
##    secax_y2 = skew.ax.secondary_yaxis('left', functions=(p2h,h2p))
##    secax_y2.set_ticks(rangeH)
####################    secax_y2.set_major_formatter(FormatStrFormatter('%.1f'))
    
    ########    STORM MOVEMENT
    right_mover,left_mover,wind_mean = mpcalc.bunkers_storm_motion(p_m, u, v, H*units.meter)
##    print('storm mover right',right_mover)
##    print('storm mover left',left_mover)
##    print('storm wind wind',wind_mean)

    SRH=mpcalc.storm_relative_helicity(H*units.meter,u,v,h1_srh * units.m)
    Ang_Crt=mpcalc.critical_angle(p_m, u, v, H*units.meter, right_mover[0], right_mover[1])
##    print(Ang_Crt,np.max(Ang_Crt))
##    print('SRH ',SRH)
    srh=float(SRH[2]/(units.meter**2/units.second**2))#valor del storm_relative_helicity 
##    print('valor buscat a 3km',float(SRH[1]/(units.meter**2/units.second**2)))
##    print(mpcalc.storm_relative_helicity((u/1.852),(v/1.852),H*units.meter,3000 * units.meter))
##    print(Stm)
    
##    print('maxim',int(np.max(VelM/1.852)))
##    print('problema',float(np.max(VelM)/units.kt))
    nearest_multiple = 5 * (1+round(int(float(np.max(VelM[:IndH])))/5))
##    print('incr',nearest_multiple)

    ax4 = fig.add_subplot(gs[0:2, 2])
    m = Basemap(llcrnrlon=-3.,llcrnrlat=39.,urcrnrlon=5.,urcrnrlat=45.,\
                rsphere=(6378137.00,6356752.3142),\
                resolution='i',projection='merc',\
                lat_0=41.,lon_0=2.)
    x, y = m(lon,lat)
    m.scatter(x,y,2,marker='o',color='r')
##    for j in range(len(x)):
##        m.plot(x[j],y[j],linewidth=0.8,color='r')

    m.drawcoastlines()
    m.fillcontinents(lake_color='#99ffff',alpha=0.4)
    # draw parallels
    m.drawparallels(np.arange(36,48,2),labels=[1,0,0,0])
    # draw meridians
    m.drawmeridians(np.arange(-4,6,2),labels=[0,0,0,1])




    
    ax2 = fig.add_subplot(gs[0:2, 3])
    Speed=mpcalc.wind_speed(wind_mean[0], wind_mean[1])
    Direc=mpcalc.wind_direction(wind_mean[0], wind_mean[1],'to')
    StmSpeed=round(float(Speed/ (units.meter / units.second)),1)#nusos respecte el vent calculat amb el srh
    StmDir=round(float(Direc/units.degree),1)#graus respecte el vent calculat amb el srh
    
    h = Hodograph(ax2, component_range=nearest_multiple)
    if nearest_multiple<=15:
        Incr=5
        amplada=1
    else:
        Incr=20
        if np.nanmax(StmSpeed)>25:
            
            amplada=2
        else:
            amplada=1
    h.add_grid(increment=Incr)
##    h.plot(u, v)
    l=h.plot_colormapped(uT, vT,PT,linewidth=5,intervals=50.,alpha=.6)  # Plot a line colored in pressure function
    plt.colorbar(l,label='Pressure')
    plt.title('nusos de 0 a '+str(h1_srh)+' km')
##    Speed=mpcalc.wind_speed(wind_mean[0], wind_mean[1])
##    Direc=mpcalc.wind_direction(wind_mean[0], wind_mean[1],'to')

    
    
    
    h.wind_vectors(wind_mean[0],wind_mean[1],width=amplada,alpha=.5)#LA LINEA SHA DE FER MES AMPLA,fico els vectors al reves, ja que covection =to

    EHI=round(float((U_CAPE*srh )/160000),2)
##    print('EHI',EHI)

    
    
    ##TROBEM LA ISOZERO
    Tz=np.transpose(T)
    Hz=np.transpose(H)
    Pz=np.transpose(P)
    idx2=np.where(np.sign(Tz[:-1]) != np.sign(Tz[1:]))[0] + 1
    ISOZP=(Pz[idx2[0]]+Pz[idx2[0]-1])/2.
    ISOZH=(Hz[idx2[0]]+Hz[idx2[0]-1])/2.
    
    
######################################    FALTA FER ELS CALCULS DELS PARÀMETRES
    ax5= fig.add_subplot(gs[2, 3])
    ax5.text(0.5, .80, 'Hod'+r'$\grave o$'+'grafa at '+str(h1_srh)+' km', fontsize=12, color='r',fontweight="bold", va="center", ha="center")
##    ax3.text(0.5, 0.95, 'EHI = '+str(EHI)+r'${m}^{2}{s}^{-2}$', fontsize=8, fontweight="bold", va="center", ha="center")
    ax5.text(0.5, 0.60, 'EH = '+('%5.1f' % (SRH[2]/ ((units.meter * units.meter) / (units.second * units.second))))+' '+r'$\mathbf{{m}^{2}{s}^{-2}}$', fontsize=8, fontweight="bold", va="center", ha="center")
    ax5.text(0.5, 0.40, 'SRH = '+('%5.1f' % (SRH[1]/ ((units.meter * units.meter) / (units.second * units.second))))+' '+r'$\mathbf{{m}^{2}{s}^{-2}}$', fontsize=8, fontweight="bold", va="center", ha="center")
    ax5.text(0.5, 0.20, 'StmDir = '+str(StmDir)+' Graus', fontsize=8, fontweight="bold", va="center", ha="center")
    ax5.text(0.5, 0.00, 'StmSpd = '+str(StmSpeed)+' Nusos', fontsize=8, fontweight="bold", va="center", ha="center")
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.axis('off')

    ax3 = fig.add_subplot(gs[3:5, 2])
    ax3.text(0.5, 0.98, 'P'+r'$\grave a$'+'rametres generals', fontsize=12, color='r',fontweight="bold", va="center", ha="center")
    ax3.text(0.5, 0.84, 'Isozero height = '+str(round(ISOZH,1))+' m', fontsize=8, fontweight="bold", va="center", ha="center")
    ax3.text(0.5, 0.70, 'LCL = '+('%5.1f' % (lcl_pressure / units.hPa))+' hPa', fontsize=8, fontweight="bold", va="center", ha="left")
    ax3.text(0.5, 0.70, 'NCC = '+str(Prat)+' hPa   ', fontsize=8, fontweight="bold", va="center", ha="right")
    ax3.text(0.5, 0.56, 'PW = '+('%5.2f' % (pw / units.mm))+' mm  ', fontsize=8, fontweight="bold", va="center", ha="right")
    ax3.text(0.5, 0.56, 'EHI = '+str(EHI), fontsize=8, fontweight="bold", va="center", ha="left")
    ax3.text(0.5, 0.42, 'TT = '+str(round(TT,0))+' Celsius', fontsize=8, fontweight="bold", va="center", ha="left")
    ax3.text(0.5, 0.42, 'K = '+str(round(Pam_K,0))+' Celsius  ', fontsize=8, fontweight="bold", va="center", ha="right")
    if np.isnan(float(lfc_pressure/units.hPa)):
        ax3.text(0.5, 0.28, 'LFC = ---- hPa', fontsize=8, fontweight="bold", va="center", ha="left")
    else:
        ax3.text(0.5, 0.28, 'LFC = '+('%5.2f' % (lfc_pressure / units.hPa))+' hPa'  , fontsize=8, fontweight="bold", va="center", ha="left")
    if np.isnan(float(EL[0]/units.hPa)):
        ax3.text(0.5, 0.28, 'EL = ---- hPa  '  , fontsize=8, fontweight="bold", va="center", ha="right")
    else:
        ax3.text(0.5, 0.28, 'EL = '+('%5.2f' % (EL[0] / units.hPa))+' hPa  ', fontsize=8, fontweight="bold", va="center", ha="right")
    ax3.text(0.5, 0.14, 'LI = '+(str(round(li,2)))+' Celsius  ', fontsize=8, fontweight="bold", va="center", ha="right")
    ax3.text(0.5, 0.14, 'T(850hPa) = '+(str(round(T[ind850],1)))+' Celsius  ', fontsize=8, fontweight="bold", va="center",  ha="left")

    ax6 = fig.add_subplot(gs[3:5, 3])
    ax6.text(0.5, 0.84, 'A '+str(Pref)+' hPa', fontsize=12, color='r',fontweight="bold", va="center", ha="center")
    ax6.text(0.5, 0.70, 'T = '+str(round(T[ind1000],1))+' Celsius ', fontsize=8, fontweight="bold", va="center", ha="right")
    ax6.text(0.5, 0.70, 'Td = '+str(round(Td[ind1000],1))+' Celsius', fontsize=8, fontweight="bold", va="center", ha="left")
    ax6.text(0.5, 0.56, r'$\theta_E$ = '+('%5.1f' % (tempot/ units.K))+' K', fontsize=8, fontweight="bold", va="center", ha="center")
##    ax3.text(0.5, 0.25, 'LI = '+('%5.1f' % ((LI_surf/units.degC)))+'Celsius  ', fontsize=8, fontweight="bold", va="center", ha="right")
##    ax3.text(0.5, 0.20, 'CAPE = '+str(S_CAPE)+' J /kg  ', fontsize=8, fontweight="bold", va="center", ha="right")
##    ax3.text(0.5, 0.20, 'CIN = '+str(S_CIN)+' J /kg', fontsize=8, fontweight="bold", va="center", ha="left")
    
    ax6.text(0.5, 0.42, 'M'+r'$\grave a$'+'xima Inestabilitat', fontsize=12, color='r',fontweight="bold", va="center", ha="center")
    ax6.text(0.5, 0.28, 'P = '+('%5.1f' % (mu[0]/ units.hectopascal))+' hPa  ', fontsize=8, fontweight="bold", va="center", ha="right")
    ax6.text(0.5, 0.28, r'$\theta_E$ = '+('%5.1f' % (tempot2/ units.K))+' K  ', fontsize=8, fontweight="bold", va="center", ha="left")
##    ax3.text(0.5, 0.0, 'LI = '+('%5.1f' % ((LI_un/units.K)))+' Celsius  ', fontsize=8, fontweight="bold", va="center", ha="right")
    ax6.text(0.5, 0.14, 'CAPE = '+str(U_CAPE)+' J /kg  ', fontsize=8, fontweight="bold", va="center", ha="right")
    ax6.text(0.5, 0.14, 'CIN = '+str(U_CIN)+' J /kg', fontsize=8, fontweight="bold", va="center", ha="left")

    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.axis('off')
    
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.axis('off')

    plt.suptitle('Radiosondatge de Barcelona de la Universitat de Barcelona a '+str(temps1))
##    mng = plt.get_current_fig_manager()
##    mng.full_screen_toggle()
    plt.savefig(fid5[:-4]+'.png',bbox_inches='tight')
    plt.close(fig)
##    plt.show()



    gmap=gmplot.GoogleMapPlotter(lat[0], lon[0], 14)
    gmap.scatter(lat, lon, 'red', size=7, marker=False)
    gmap.draw(fid5[:-4]+'.html')




##############part per autoimatatizar la lectura############
np.warnings.filterwarnings('ignore')#to avoid the error messages

print('Please, insert the path where there the txt files are -- for instance C:/Soundings/Data')
Fid=input()
##Fid='c:/soundings'
##Fid='D:/Soundings_UB/'

d=glob.glob(Fid+'/'+'*.txt')
d=sorted(d)

for a in d:
    
    print('Processing file ',a)
    ReadFile(a)

