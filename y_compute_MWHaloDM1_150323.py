# M Lovell 15/03/23. This set of routines is written with the ultimate goal of computing 
# the dark matter decay intensity profile of the Milky Way dark matter halo as observered 
# from Earth. The primary routine to be called is compute_MWHaloDM. It has one input text
# file and writes two png plots to disk: dark matter density profiles and angular decay
# intensity profiles. It also writes some of the output arrays to disk so that they can
# be re-read later, thus avoiding running the computation more often than is necessary.  

# This code as originally written in IDL. It has been cleaned and functions correctly. 
# However, it is part of a live science project and has not reached its final form. Some 
# the previous IDL versions include ~6 plots, only two of which remain here. However,
# some of the variables used in the removed plots have been retained for the sake of
# expediency.

# This function computes the density, Dens, at a position dcen from the halo centre. The
# name of the required density profile, of which there are four, is provided in dproftype
# and the parameters of the profile are provided in the params variable.  
def dens_profs(dproftype, dcen, params):

    import numpy as np
    
    if dproftype == 'gNFW':
        # generalised NFW profile
  
        rs = params[0] ; rhos = params[1] ; gam1 = params[2] ; gam2 = params[3]
        Dens = rhos/\
               (pow(dcen/rs,gam1)*pow(1.+dcen/rs,gam2))
       
    elif dproftype == 'Einasto': 
        # Einasto profile

        rs = params[0] ; rhos = params[1] ; alp = params[2] 
        Dens = rhos*np.exp(\
                           -2/alp*(pow(dcen/rs,alp)-1))
                    
    elif dproftype == 'Burkert': 
        # Burkert profile
     
        rc = params[0] ; rhos = params[1] 
        Dens = rhos/ \
               ((1+dcen/rc)*pow(1+(dcen/rc),2))     

    elif dproftype == 'UniSlab': 
        # Uniform slab for testing / debugging purposes. Not applied in this version of the code.

        redge = params[0] ; rhos = params[1] ; bet = params[3] 
        if abs(dcen*math.cos(bet)) < redge :
            Dens = rhos
        else :
            Dens = 0
    else : 
        # If the value of dproftype does not match any of the 4 options, say so.
        print('None of the above!')

    return Dens

# This routine computes the projected dark matter / decay flux for a given observation
# angle and dark matter profile. The profile type and its parameters are given by
# dproftype and params respectively. The distance of the observer from the MW 
# centre is given by dSMWC, the field of view of the telescope by fovr, and the angle 
# from the centre of the halo. The integration is performed from a minimum distance
# -- mindist -- to a maximum distance -- maxdist -- across a number of steps given by nstep.
# It returns six variables: three fluxes and three projected densities. Flx is the flux measured
# using the stated dproftype and its parameters, Flxc, the flux when the P17 core is
# imposed, and Flxcore is the flux from the core alone. Sdm, Sdmc, and Sdmcore
# are the corresponding projected densities.

def compute_sightline(dproftype, fovr, dSMWC, params,\
                      ang, mindist, maxdist, nstep):

    import numpy as np
    
    # Define the parameters for this dproftype profile
    params1 = params[0:4]
    
    # Define the parameters for the P17 core
    paramsP17 = params[4:7]

    # Establish the distance integration steps, measured from the observer
    # and the thickness of each slice / dark matter packet
    dsteps = np.zeros(nstep, dtype=float)
    for i in range(0,nstep):
        dsteps[i] = i*(maxdist-mindist)/(nstep-1)+mindist
    dthick = dsteps[1]-dsteps[0]

    # Initialise the three fluxes and three projected densities at 0
    Sdm     = 0.
    Sdmc    = 0.
    Sdmcore = 0.

    Flx     = 0.
    Flxc    = 0.
    Flxcore = 0.

    #Integrate over all the distance from the observer steps
    for i in range(0, nstep):

        # Compute the distances to the dark matter packet from the
        # halo centre as projected on the line that joins the observer 
        # and the Milky Way centre
        dca = dsteps[i]*abs(np.cos(ang))
        dsa = dsteps[i]*np.sin(ang)

        # Compute the distance from the packet to the Milky Way centre
        if ang < np.pi/2 : 
            dcen = np.sqrt(pow(dca-dSMWC,2)+pow(dsa,2))
        else : 
            dcen = np.sqrt(pow(dca+dSMWC,2)+pow(dsa,2))

        # Compute the volume of the packet
        vol = np.pi*pow(dsteps[i]*fovr/2,2)*dthick

        # Call dens_profs to compute the packet density at the distance.
        # The variable dens is this profile's density, and densc=dens if
        # dens is less than the P17 core density
        Dens = dens_profs(dproftype, dcen, params1)
        Densc = Dens

        # Compute the density of the P17 profile if dcen<3kpc
        if dcen < 3 and paramsP17[0] > 0. : 
            Dense = dens_profs('Einasto', dcen, paramsP17)

            # Impose this core density of dense if it is less than
            # Dens
            if Dense < Densc:
                Densc = Dense

        # Compute the additions to the projected densities and fluxes
        Sdmcore += Densc*dthick
        Flxcore += Densc*vol/(4*np.pi*pow(dsteps[i],2))
        Sdm = Sdm + Dens*dthick
        Sdmc = Sdmc + Densc*dthick
        Flx = Flx + Dens*vol/(4*np.pi*pow(dsteps[i],2))
        Flxc = Flxc + Densc*vol/(4*np.pi*pow(dsteps[i],2))

    return Flx, Flxc, Flxcore, Sdm, Sdmc, Sdmcore

# The main program. Initialises various model parameters and fit parameters,
# call routines that compute the density profiles and the subsequent 
# decay profiles. Plots the results. 
def compute_MWHaloDM(void):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.axis as ax

    print('Starting')

    ######################################
    # Part 1, initialise constants

    # Initialise constants, including universal gravitational constant ugrav,
    # proton mass in kg mprkg, solar mass in kg msunkg, and the kpc in cm kpcincm
    ugrav  = 4.302e-6
    mprkg  = 1.66e-27
    msunkg = 2.0e30 
    kpcincm = 3.086e21

    # Initialize sterile neutrino parameters including mass mnu
    # mixing angle s2t, and the subsequent decay rate gam.
    # Also the proton mass mpr.
    is2t = 5
    s2t = is2t*1e-11
    s2tstr = 'sin^{2}(2\theta)='+str(is2t)+'x10^{-11}'
    mnu = 7.1
    mpr = 938e3
    gam = 1.38e-29*s2t/1e-7*pow(mnu,5) # Bulbul+14 Eqn1

    # Milky Way halo parameters, distance to Milky Way centre dSMWC
    # and rhoSMW 
    dSMWC = 8.13e0 # Dessert+20 value
    rhoSMW = 400e3/mpr*(mprkg/msunkg)*pow(kpcincm,3)


    # Field of view for the measurement in arcmin, fovam. Two options: XMM-Newton and the LW. 
    # Then compute the FoV in radians, fovr, and the solid angle, ster.

    #  fovam = 23. # XMM-Newton
    fovam = np.sqrt(66./np.pi) # Approximate LW in Hofmann, from the sqrt of its area
    fovr = fovam/60e0*(np.pi/180)
    ster = 2.*np.pi*(1e0-np.cos(fovr))
    print('Fov:', "%.2f" % fovam, "%.2e" % fovr, "%.2e" % ster)

    # Compute the two normalisations, to compute the X-ray flux: dfac and dfacSDM. 
    # The dfacsdm equation comes from Hofmann Eqn 1, to multiply Sdm 
    dfac = gam/(pow(kpcincm,2)*((mnu/mpr)*mprkg/msunkg))
    dfacsdm = ster*\
              0.1*(s2t/1e-11)*(1e0/3.25e0)*(1e0/1e9)*pow(mnu/7,4) 

    # Save options to write binary data to disk and not have to recompute
    # on all runs. SavFile stores the name of the save file, and compsight
    # is the toggle to do the computation (compsight=1) or to load the previous
    # data (compsight!=1)
    SavFile = 'save_fluxes'
    compsight = 0


    ######################################
    # Part 2, parameters for density profiles

    # Initialise parameters for the reported density profiles. rsX,rcX==scale radius,
    # rhosX==characteristic density, alpX==shape parameter, gam1X==first power law,
    # gam2X==second power law, where X is the given profile parameters. Write two sets of 
    # parameters, one for X -- paramsX -- and and another that also includes P17 
    # (see below) for including the central density core, paramsXc.

    # Portail+ 2017 (P17) Einasto fit to the MW centre, <3kpc. Einasto profile, used in Hofmann+19.
    # Ignore the P17 flattening for now. Have dummy data for paramsP17c.

    rsP17   =  7.1e0 
    rhosP17 =  1.8e7 
    alpP17  =  0.77e0
    paramsP17 = [rsP17,rhosP17,alpP17]
    paramsP17c  = np.zeros(7)
    paramsP17c[0:3] = paramsP17
    paramsP17c[3:7] = [-1.,-1.,-1.,-1.]

    # McMillan+ 2017 (M17) Table 3. NFW fit. Used in Boyarsky+ 2018.
    rsM17 = 19.6e0 
    rhosM17 = 8.5e6
    gam1M17 = 1.
    gam2M17 = 2.  
    paramsM17  = [rsM17, rhosM17, gam1M17, gam2M17]
    paramsM17c  = np.zeros(7)
    paramsM17c[0:4] = paramsM17
    paramsM17c[4:7] = [rsP17,rhosP17,alpP17]

    # NFW fit used in Dessert+ 20
    rcD20n = 20. # NFW profile, Dessert+ 20
    rhosD20n = rhoSMW*\
               (dSMWC/rcD20n)*pow(1+dSMWC/rcD20n,2)    
    paramsD20n  = [rcD20n, rhosD20n, 1, 2]
    paramsD20nc  = np.zeros(7)
    paramsD20nc[0:4] = paramsD20n
    paramsD20nc[4:7] = [rsP17,rhosP17,alpP17]

    # Burkert profile also used in Dessert+ 20 S17
    rcD20b = 9.  
    rhosD20b = rhoSMW*\
               ((1.+dSMWC/rcD20b)*pow(1.+(dSMWC/rcD20b),2))
    paramsD20b  = [rcD20b, rhosD20b, 1, 2]
    paramsD20bc  = np.zeros(7)
    paramsD20bc[0:4] = paramsD20b
    paramsD20bc[4:7] = [rsP17,rhosP17,alpP17]
    
    # Cautun+ 2020 circular velocity profile, digitized from 
    # Cautun+ 2020 (C20). Read the file and use to derive a 
    # series of NFW profiles with the 
    # same mass. This will show how C20 is contracted compare to NFW.

    # Read the file with the C20 mass profile data in it, store as strings
    # in C20arrstrs and get the number of lines as nl0
    fname = 'Cautun2020DarkMatterData.dat'
    f = open(fname, 'r')
    C20arrstrs = f.readlines()
    nl0 = len(C20arrstrs)
    
    # Read in log10 radius lrad and circular velocity vcC20DM from string,
    # then fill up the radius array radarr and enclosed mass array
    # meC30DMarr

    radarr = np.zeros(nl0)
    meC20DMarr = np.zeros(nl0)
    
    for i in range(0,nl0):
        strs = C20arrstrs[:][i].split(' ')
        lrad = float(strs[0])
        vcC20DM = float(strs[2])
        radarr[i] = pow(10.,lrad)
        meC20DMarr[i] = pow(vcC20DM,2)*radarr[i]/ugrav
        

    # Write a series of nrs=10 NFW profiles with scale radii logarithmically
    # spaced between lower and uppwer bounds rsmin and rsmax.
    rsmin = 5. ; rsmax = 80. ; nrs = 10  
    rsarr = np.zeros(nl0, dtype=float)
    for i in range(0, nrs) :
        rsarr[i] = pow(10,i*(np.log10(rsmax)-np.log10(rsmin))/(1*nrs-1)+np.log10(rsmin))
  
    # Compute the characteristic densities, rhosarr, for these NFW profiles, using the total
    # mass enclosed in C20.
    rhosarr = np.zeros(nrs,dtype=float)
    for i in range(0,nrs) : 
        rrat = radarr[nl0-1]/rsarr[i]
        rhosarr[i] = meC20DMarr[nl0-1]/\
                     (4.*np.pi*pow(rsarr[i],3)*(np.log(1.+rrat)-1./(1./rrat+1.)))

    # C20 Density profile parameters, estimated elsewhere from digitizing the C20 figure. 
    rsC20 = 20.
    rhosC20 = 3e6
    gam1C20 = 1.57 
    gam2C20 = 1.13
    paramsC20  = [rsC20, rhosC20, gam1C20, gam2C20]
    paramsC20c  = np.zeros(7)
    paramsC20c[0:4] = paramsC20
    paramsC20c[4:7] = [rsP17,rhosP17,alpP17]

    # Compute the five density profiles used / adapted from observational studies.
    rhoarrD20n = dens_profs('gNFW', radarr, paramsD20n)
    rhoarrD20b = dens_profs('Burkert', radarr, paramsD20b)    
    rhoarrM17 = dens_profs('gNFW', radarr, paramsM17)
    rhoarrC20 = dens_profs('gNFW', radarr, paramsC20)
    rhoarrP17 = dens_profs('Einasto', radarr, paramsP17)

    #Define the colours associated with the models
    colM17 = 'gray'
    colC20 = 'black'
    colP17 = 'brown'
    colD20n = 'turquoise'
    colD20b = 'turquoise'

    ######################################
    # Part 3, plot density profiles

    # Get colours for the nrs density profiles, using hexadecimal to define colours
    # from blue to red. Store in list clval
    col = []
    for i in range(0,nrs):
        valr = int(255*(i+1)/(1*nrs))
        valb = int(255*(nrs-i)/(1*nrs))
        clval = '#'+hex(valr).lstrip("0x")+'00'+hex(valb).lstrip("0x")
        col.append(clval)

    # Plot lines used in previous papers. P17 was only considered for distances<3kpc,
    # so we will draw this line as solid within 3kpc and dotted outside it.
    # Also, D20 considered two lines. We will show their NFW line as solid and their
    # Burkert as dashed.
    plt.plot(radarr, rhoarrM17, color=colM17,linewidth=3.) 
    plt.plot(radarr, rhoarrC20, color=colC20,linewidth=3.) 
    plt.plot(radarr, rhoarrP17, color=colP17,linewidth=3.,linestyle='dotted')
    i1 = np.where(radarr < 3.)
    plt.plot(radarr[i1], rhoarrP17[i1], color=colP17,linewidth=3.)
    plt.plot(radarr, rhoarrD20n, color=colD20n,linewidth=3.)
    plt.plot(radarr, rhoarrD20b, color=colD20b,linewidth=3.,linestyle='--')   
    

    # Compute the nrs density profiles and plot these
    for i in range(0,nrs):
        params = [rsarr[i], rhosarr[i],1,2]
        rhoarr = dens_profs('gNFW', radarr, params)
        plt.plot(radarr, rhoarr, color=col[i],linewidth=1.) 

    # Set axes parameters and the legend, (previous paper lines, ignore nrs
    # lines for now) then save to disk as DensProfsP.png. Comment out
    # plt.show(), may uncomment later if we wish.
    plt.axis([0.8,200.,3e3,1e9])
    ax = plt.gca()
    ax.tick_params(tickdir='in',which='both',labelsize=12)

    plt.xlabel(r'$r~\mathrm{[kpc]}$',fontsize=14)
    plt.ylabel(r'$\rho~\mathrm{[M_{\odot}/kpc^{3}]}$', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')

    plt.legend(['M17','C20','P17','P17c','D20','D20b'])

    #plt.show()
    plt.savefig('./DensProfsP.png')
    plt.close()


    ######################################
    # Part 4, compute angular decay profiles
    #
    # Computing the sightlines currently takes ~20 minutes to run. 
    # If the toggle compsight=1 the code will do the computation and write the 
    # results to disk. Any other value of compsight will reread the results from disk,
    # which is useful for times that further analysis needs to be performed / plots
    # need to be reproduced.


    if compsight == 1 :

        # Define a range of observation angles, where 0deg is directly towards
        # the Milky Way Centre and 180deg is directly away. We will compute
        # nang angles in the range maxang=180deg to minang = 0.5deg, converting
        # to radians and storing in angarr.    
        maxang = np.pi ; minang = 0.5*np.pi/180 ; nang = 360
        angarr = np.zeros(nang)
        for i in range(0,nang) :
            angarr[i] = i*(maxang-minang)/(nang*1. - 1)+minang
    
        # Set the range of distances from the observer over which to integrate
        # the sightline. We will take number of steps=nstep between a 
        # minimum distance mindist and maximum distance maxdist, both in kpc
        maxdist = 190. ; nstep = 10000 ; mindist = 0.05
    
        # Initialise arrays for the various flux and projected mass profiles
        # of the nrs NFW profiles. We also define arrays to compute the
        # the fraction of the decay / projected density that is with the 
        # Milky Way core, as Fraccorea and Fraccoresdma. Note that Fraccorea
        # and Fraccoresdma are calculated but not otherwise used in this version
        # of the code.
        Flxa = np.zeros(nrs*nang)
        Flxa = Flxa.reshape(nrs,nang)
        Flxca = np.zeros(nrs*nang)
        Flxca = Flxca.reshape(nrs,nang)
        Fraccorea = np.zeros(nrs*nang)
        Fraccorea = Fraccorea.reshape(nrs,nang)    

        Sdma = np.zeros(nrs*nang)
        Sdma = Sdma.reshape(nrs,nang)
        Sdmca = np.zeros(nrs*nang)
        Sdmca = Sdmca.reshape(nrs,nang)
        Fraccoresdma = np.zeros(nrs*nang)
        Fraccoresdma = Fraccoresdma.reshape(nrs,nang)

        # Compute the angular decay profiles of the nrs profiles.
        # For each nrs profile we store the parameters in rs and rhos,
        # then cycle over the nang angles and call compute sightline to 
        # obtain the fluxes and projected densities. We then store the fluxes
        # and the core fractions in the defined arrays.
        gam1NFW = 1 ; gam2NFW = 2
        for i in range(0,nrs):
            rs = rsarr[i]
            rhos = rhosarr[i]
            alp = -1
            print('Computing NFW intensity profile: ', "i=%i" % i, "of nrs=%i;" % nrs,\
                  "rs=%.2f," % rs, "rhos=%.2e," % rhos, "gam1=%.2f," % gam1NFW, " and gam2=%.2f" % gam2NFW)
            for j in range(0,nang):
                ang = angarr[j]          
                params = [rs, rhos, gam1NFW, gam2NFW, rsP17, rhosP17, alpP17]
                
                Flx, Flxc, Flxcore, Sdm, Sdmc, Sdmcore = compute_sightline('gNFW', fovr, dSMWC, params,\
                                                                           ang, mindist, maxdist, nstep)
                Flxa[i,j] = Flx*dfac  
                Flxca[i,j] = Flxc*dfac
                Fraccorea[i,j] = Flxcore/Flxc             
                Sdma[i,j] = Sdm  
                Sdmca[i,j] = Sdmc
                Fraccoresdma[i,j] = Sdmcore/Sdmc               


        # We now compute the flux / projected density profiles for the C20 profile in 
        # the same manner as for the nrs profiles.
        FlxC20 = np.zeros(nang)
        FlxcC20 = np.zeros(nang)
        FraccoreC20 = np.zeros(nang)
        
        SdmC20 = np.zeros(nang)
        SdmcC20 = np.zeros(nang)
        FraccoresdmC20 = np.zeros(nang)

        print('Computing C20 intensity profile: ',\
              "rs=%.2f," % rsC20, "rhos=%.2e," % rhosC20, "gam1=%.2f," % gam1C20, " and gam2=%.2f" % gam2C20)
        for j in range(0,nang) :
            ang = angarr[j]        
            Flx, Flxc, Flxcore, Sdm, Sdmc, Sdmcore = compute_sightline('gNFW', fovr, dSMWC, paramsC20c,\
                                                                       ang, mindist, maxdist, nstep)
            FlxC20[j] = Flx*dfac
            FlxcC20[j] = Flxc*dfac
            FraccoreC20[j] = Flxcore/Flxc
            SdmC20[j] = Sdm 
            SdmcC20[j] = Sdmc 
            FraccoresdmC20[j] = Sdmcore/Sdmc

        # We now compute the flux / projected density profiles for the M17 profile in 
        # the same manner as for the nrs profiles.
        FlxM17 = np.zeros(nang)
        FlxcM17 = np.zeros(nang)
        FraccoreM17 = np.zeros(nang)
        
        SdmM17 = np.zeros(nang)
        SdmcM17 = np.zeros(nang)
        FraccoresdmM17 = np.zeros(nang)
        
        print('Computing M17 intensity profile: ',\
          "rs=%.2f," % rsM17, "rhos=%.2e," % rhosM17, "gam1=%.2f," % gam1M17, " and gam2=%.2f" % gam2M17)    
        for j in range(0, nang) :
            ang = angarr[j]        
            Flx, Flxc, Flxcore, Sdm, Sdmc, Sdmcore  = compute_sightline('gNFW', fovr, dSMWC, paramsM17c,\
                                                                        ang, mindist, maxdist, nstep)
            FlxM17[j] = Flx*dfac
            FlxcM17[j] = Flxc*dfac
            FraccoreM17[j] = Flxcore/Flxc
      
            SdmM17[j] = Sdm
            SdmcM17[j] = Sdmc
            FraccoresdmM17[j] = Sdmcore/Sdmc
      
        # We now compute the flux / projected density profiles for the P17 profile in 
        # the same manner as for the nrs profiles.
        FlxP17 = np.zeros(nang)
        FlxcP17 = np.zeros(nang)
        FraccoreP17 = np.zeros(nang)
        SdmP17 = np.zeros(nang)
        SdmcP17 = np.zeros(nang)
        FraccoresdmP17 = np.zeros(nang)

        print('Computing P17 intensity profile: ',\
              "rs=%.2f," % rsP17, "rhos=%.2e," % rhosP17, " and alp=%.2f" % alpP17)
        for j in range(0,nang) :
            ang = angarr[j]        
            Flx, Flxc, Flxcore, Sdm, Sdmc, Sdmcore = compute_sightline('Einasto', fovr, dSMWC, paramsP17c,\
                                                                       ang, mindist, maxdist, nstep)
            FlxP17[j] = Flx*dfac
            FlxcP17[j] = Flxc*dfac
            FraccoreP17[j] = Flxcore/Flxc
            
            SdmP17[j] = Sdm
            SdmcP17[j] = Sdmc
            FraccoresdmP17[j] = Sdmcore/Sdmc

        # Save the results to SavFile
        np.savez(SavFile,\
                 Flxa=Flxa, Flxca=Flxca, Fraccorea=Fraccorea,\
                 Sdma=Sdma, Sdmca=Sdmca, Fraccoresdma=Fraccoresdma,\
                 FlxC20=FlxC20, FlxcC20=FlxcC20, FraccoreC20=FraccoreC20,\
                 SdmC20=SdmC20, SdmcC20=SdmcC20, FraccoresdmC20=FraccoresdmC20,\
                 FlxM17=FlxM17, FlxcM17=FlxcM17, FraccoreM17=FraccoreM17,\
                 SdmM17=SdmM17, SdmcM17=SdmcM17, FraccoresdmM17=FraccoresdmM17,\
                 FlxP17=FlxP17, FlxcP17=FlxcP17, FraccoreP17=FraccoreP17,\
                 SdmP17=SdmP17, SdmcP17=SdmcP17, FraccoresdmP17=FraccoresdmP17,\
                 angarr=angarr)
    else :
        # Read in previously compute arrays from SavFile
        print('Reading in previously computed arrays from '+SavFile+'.npz')
        data = np.load(SavFile+'.npz')
        Flxa=data['Flxa'] ;  Flxca=data['Flxca'] ;  Fraccorea=data['Fraccorea']  
        Sdma=data['Sdma'] ;  Sdmca=data['Sdmca'] ;  Fraccoresdma=data['Fraccoresdma']  
        FlxC20=data['FlxC20'] ;  FlxcC20=data['FlxcC20'] ;  FraccoreC20=data['FraccoreC20']  
        SdmC20=data['SdmC20'] ;  SdmcC20=data['SdmcC20'] ;  FraccoresdmC20=data['FraccoresdmC20']  
        FlxM17=data['FlxM17'] ;  FlxcM17=data['FlxcM17'] ;  FraccoreM17=data['FraccoreM17']  
        SdmM17=data['SdmM17'] ;  SdmcM17=data['SdmcM17'] ;  FraccoresdmM17=data['FraccoresdmM17'] 
        FlxP17=data['FlxP17'] ;  FlxcP17=data['FlxcP17'] ;  FraccoreP17=data['FraccoreP17']  
        SdmP17=data['SdmP17'] ;  SdmcP17=data['SdmcP17'] ;  FraccoresdmP17=data['FraccoresdmP17']
        angarr=data['angarr']


    print('Finished computing angular profiles')

    ######################################
    # Part 5, plot angular decay profiles

    # Find angarr in degrees, angarrd, and plot five of the paper
    # density profile-drived angular profiles. We will skip the nrs
    # profiles for this plot. 
    angarrd = angarr*180./np.pi
    plt.plot(angarrd, FlxM17/ster, color=colM17,linewidth=3.) 
    plt.plot(angarrd, FlxC20/ster, color=colC20,linewidth=3.) 
    plt.plot(angarrd, FlxP17/ster, color=colP17,linewidth=3.)
    plt.plot(angarrd, FlxcM17/ster, color=colM17,linewidth=3.,linestyle='--') 
    plt.plot(angarrd, FlxcC20/ster, color=colC20,linewidth=3.,linestyle='--') 

    # Plot data point from Hofmann+2019. This is not referenced in the legend.
    plt.plot([1.5,1.5],[1e-7/ster,8e-7/ster], marker='_',\
             color='green',linewidth=1.5,markersize=10)
    plt.plot([1.5],[4.5*1e-7/ster], marker='o',\
             color = 'green', markersize=10)

    # Set axes parameters and the legend, (previous paper lines, ignore nrs
    # lines for now) then save to disk as IntensThetaP.png. Comment out
    # plt.show(), may uncomment later if we wish.
    plt.axis([-5,180.,1.5e-3,3e-1])
    ax = plt.gca()
    ax.tick_params(tickdir='in',which='both',labelsize=12)

    plt.xlabel(r'$\theta_{GC}~[deg.]$',fontsize=14)
    plt.ylabel(r'$I~[cts~sec^{-1}str^{-1}cm^{-2}]$', fontsize=14)
    plt.yscale('log')

    plt.legend(['M17','C20','P17','M17c','C20c'])

    #plt.show()
    plt.savefig('./IntensThetaP.png')
    plt.close()

    print('Finished all plots and computations')
    # End of code



  
  
  
