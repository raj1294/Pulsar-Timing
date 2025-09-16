import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob, os
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS

ks = 1000

##Pipeline to construct lightcurves for pulsation searches from XMM catalogue 
#of Walton et al. (2022)
fname = "walton_catalogue.fits"
hdulist = fits.open(fname)
data = hdulist[1].data

#Relevant catalogue parameters
obsid = data['ObsID']
ra = data['RADdeg']
dec = data['DEDdeg']
galaxies = data['Name']
counts = data['EP-8cts']
pnfvar = data['PN-FVar']
lumin = data['Lum8']

#Set selection criteria
ctthresh = 4000
galaxies = galaxies[counts>ctthresh]
pnfvar = pnfvar[counts>ctthresh]
obsid = obsid[counts>ctthresh]
ra = ra[counts>ctthresh]
dec = dec[counts>ctthresh]
lumin = lumin[counts>ctthresh]
counts = counts[counts>ctthresh]

#Add Holmberg IX X-1 and NGC 1313 X-2 to the sample
obsid = list(obsid)
galaxies = list(galaxies)
ra = list(ra)
dec = list(dec)
counts = list(counts)
lumin = list(lumin)
pnfvar = list(pnfvar)
obsid_unique_ = np.unique(obsid)
obsunique,galaxies_unique = [[],[]]
for i in range(len(obsid_unique_)):
    for i2 in range(len(obsid)):
        if(obsid_unique_[i]==obsid[i2]):
            obsunique.append(obsid_unique_[i])
            galaxies_unique.append(galaxies[i2])
            break

for i3 in range(len(obsunique)):    
    
    if(galaxies_unique[i3]=='NGC1313'):
        
        ra.append(49.5924141)
        dec.append(-66.6009329)
        galaxies.append('NGC1313')
        obsid.append(obsunique[i3])
        pnfvar.append(0)
        lumin.append(1e40)
        counts.append(10000)
        
    if(galaxies_unique[i3]=='NGC3031'):
        
        ra.append(149.4720417)
        dec.append(69.0633833)
        galaxies.append('NGC3031')
        obsid.append(obsunique[i3])
        pnfvar.append(0)
        lumin.append(1e40)
        counts.append(10000)
    
    if(galaxies_unique[i3]=='NGC3621'):
                
        ra.append(169.5625817)
        dec.append(-32.8107594)
        galaxies.append('NGC3621')
        obsid.append(884030101)
        pnfvar.append(0)
        lumin.append(1e40)
        counts.append(10000)
    
    if(galaxies_unique[i3]=='NGC4559'):
                
        ra.append(188.9649712)
        dec.append(27.9345831)
        galaxies.append('NGC4559')
        obsid.append(842340201)
        pnfvar.append(0)
        lumin.append(1e40)
        counts.append(10000)
        
        ra.append(188.9649712)
        dec.append(27.9345831)
        galaxies.append('NGC4559')
        obsid.append(883960201)
        pnfvar.append(0)
        lumin.append(1e40)
        counts.append(10000)
    

obsid,galaxies,ra,dec,counts,lumin,pnfvar =\
    zip(*sorted(zip(obsid,galaxies,ra,dec,counts,lumin,pnfvar)))

obsid_unique,nrepeats = np.unique(obsid,return_counts=True)
nrep,gal_unique = [[],[]]
for i4 in range(len(obsid_unique)):
    
    #Deal with repeat ObsIDs (due to multiple sources)
    nrepindex = 0
    for i5 in range(len(obsid)):
        if(obsid[i5]==obsid_unique[i4]):
            nrep.append(nrepindex)
            nrepindex += 1
    
    #Identify unique galaxies
    for i6 in range(len(obsid)):
        if(obsid[i6]==obsid_unique[i4]):
            gal_unique.append(galaxies[i6])
            break
        
ra = np.array(ra)
dec = np.array(dec)
obsid = np.array(obsid)
counts = np.array(counts)
lumin = np.array(lumin)
pnfvar = np.array(pnfvar)
galaxies = np.array(galaxies)
nrep = np.array(nrep)

#Source detection parameters
string,string2,string3,string3b,string4,string5 = [[],[],[],[],[],[]]

pimin = 300 #Min channel of image
pimax = 10000 #Max channel of image
sigthresh = 1e-10 #Source detection likelihood threshold
pixscale = 0.05 #Pixel scale (arcsec per pixel: EPIC-PN)
septhresh = 1e-2 #Sep between catalogue ULX position and detection (in deg)
dsep = 1e-5 #Sep increments
ctthreshwav = 200 #CT threshold for wavdetect

#Lightcurve parameters
bintime = 0.07336496 #LC bintime
counter = 0
counter2 = 0

tstart1,dtseg1,obsid1,evnum1 = [[],[],[],[]]

for j in range(len(obsid_unique)):
        
    if(gal_unique[j]=='NGC1313'):
                    
        #Download
        ObsId = '0' + str(obsid_unique[j])
        outfile = str(gal_unique[j]) + "_" + str(ObsId) + ".tar"
        url =\
        "'https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno="+\
        str(ObsId)+"&level=ODF'"   
        
        comm0a = "mkdir " + gal_unique[j]
        comm0b = "cd " + gal_unique[j]
        comm1a = "wget -O " + outfile + " " + url
        comm1b = "cd ../"
        
        string.append(comm0a)
        string.append(comm0b)
        string.append(comm1a)
        string.append(comm1b)
        
        #Reduce
        comm2 = "cd " + gal_unique[j]
        comm3 = "mkdir " + ObsId
        comm4 = "mkdir " + ObsId + "/odf/"
        comm5 = "mv *" + ObsId + "*.tar " + ObsId + "/odf/"
        comm6 = "cd " +  ObsId + "/odf/"
        comm7 = "tar -xvf *.tar"
        comm8 = "rm -f *.tar"
        comm9 = "tar -xvf *.TAR"
        comm10 = "rm -f *.TAR"
        comm11 = "cd ../"
        comm12 = "z=$(pwd)"
        comm13 = "rm -rf pps"
        comm14 = "mkdir proc"
        comm15 = "cd proc/"
        comm16 = "export SAS_CCF=$z" + "/odf"
        comm17 = "export SAS_ODF=$z" + "/odf"
        comm18 = "cifbuild"
        comm19 = "z=$(pwd)"
        comm20 = "export SAS_CCF=$z" + "/ccf.cif"
        comm21 = "odfingest"
        comm22 = "q=(*.SAS)"
        comm23 = "export SAS_ODF=$z" + "/$q"
        comm24 = "epproc"
        comm25 = "cd ../../../"
        comm26 = ""
                
        string2.append(comm2)
        string2.append(comm3)
        string2.append(comm4)
        string2.append(comm5)
        string2.append(comm6)
        string2.append(comm7)
        string2.append(comm8)
        string2.append(comm9)
        string2.append(comm10)
        string2.append(comm11)
        string2.append(comm12)
        string2.append(comm14)
        string2.append(comm15)
        string2.append(comm16)
        string2.append(comm17)
        string2.append(comm18)
        string2.append(comm19)
        string2.append(comm20)
        string2.append(comm21)
        string2.append(comm22)
        string2.append(comm23)
        string2.append(comm24)
        string2.append(comm25)
        string2.append(comm26)
        
        #Create image
        comm27 = "cd " + gal_unique[j] + "/" + ObsId + "/proc"
        comm28 = "d=0"
        comm29 = "for j in *ImagingEvts*.ds"
        comm30 = "do"
        comm31 = 'let "d++"'
        comm32 = "cp $j epn$d.fits"
        comm33 = "evselect table=epn$d.fits withimageset=yes imageset=" +\
                  "EPNimage$d.fits xcolumn=X ycolumn=Y" +\
                  " imagebinning=imageSize ximagesize=600 yimagesize=600"
        comm33b = "done"
        comm34 = "cd ../../../"
        comm35 = ""
        
        string3.append(comm27)
        string3.append(comm28)
        string3.append(comm29)
        string3.append(comm30)
        string3.append(comm31)
        string3.append(comm32)
        string3.append(comm33)
        string3.append(comm33b)
        string3.append(comm34)
        string3.append(comm35)
                        
        nreparr = np.arange(1,nrepeats[j]+1,1)
        
        for j2 in range(len(nreparr)):
            ctfm = 0                                   
            for fimds in glob.glob(gal_unique[j] + "/" + ObsId +\
                                   "/proc/" + "*ImagingEvts*.ds"):
                                
                ctfm += 1
                
                #Run source detection algorithm
                imfile = "EPNimage" + str(ctfm) + ".fits"
                
                hdufile = fits.open(fimds)
                tbdata = hdufile[1].data
                tbheader = hdufile[1].header
                tstart_obsid = tbheader['TSTART']
                telapse = tbheader['ONTIME']
                                        
                comm36 = "cd " + gal_unique[j] + "/" + ObsId + "/proc/"
                
                comm37 = "source activate ciao-4.17"
                comm38 = "wavdetect " + imfile + " " +\
                          "source_list.fits source_cell.fits image.fits" +\
                          " background.fits " +\
                          'expfile=none psffile=none scales="1 2 4 8 16"' +\
                          ' sigthresh=' +\
                          str(sigthresh) + ' regfile=. clobber=yes'
                comm39 = "mv source_list.fits " + "source_list" +\
                          "_obs" + str(ctfm) + "_src" + str(nreparr[j2]) +\
                          ".fits"
                comm40 = "echo $PWD"
                comm41 = "conda deactivate"
                comm45 = "cd ../../../"
                comm46 = ""
    
                comm41b = "rm -f epn" + str(ctfm) + "cp.fits"                
                comm44 = "cd ../../"
                
                string5.append(comm27)
                string5.append(comm41b)
                string5.append(comm44)
                
                string3b.append(comm36)
                string3b.append(comm37)
                string3b.append(comm38)
                string3b.append(comm39)
                string3b.append(comm40)
                string3b.append(comm41)
                string3b.append(comm45)
                string3b.append(comm46)

np.savetxt("download.sh",string,fmt='%s',delimiter='   ')
os.system("chmod u+x download.sh")
np.savetxt("reduce.sh",string2,fmt='%s',delimiter='   ')
os.system("chmod u+x reduce.sh")
np.savetxt("image.sh",string3,fmt='%s',delimiter='   ')
os.system("chmod u+x image.sh")
np.savetxt("srcdetect.sh",string3b,fmt='%s',delimiter='   ')
os.system("chmod u+x srcdetect.sh")

#Produce filtered event file and lightcurves 
Nseg = 0
for k in range(len(obsid)):
        
    ObsId = '0' + str(obsid[k])
            
    raulx = ra[k]
    deculx = dec[k]
    culx = SkyCoord(raulx,deculx,unit="deg",frame='icrs')    
    nrepind = str(nrep[k]+1)
    
    if(galaxies[k]=='NGC1313'):
    
        direc = galaxies[k] + "/" + ObsId + "/odf/"
        srclistkey = "source_list*_src" + nrepind + ".fits"
        directory = galaxies[k] + "/" + ObsId + "/proc/" + srclistkey
                  
        for srcfile in sorted(glob.glob(directory)):
            
            obsindex = srcfile.split("_obs")[1].split("_")[0]
                        
            hdu = fits.open(srcfile)
            tbdata = hdu[1].data
            tbheader = hdu[1].header
            RA = tbdata['RA']
            DEC = tbdata['DEC']
            xpos = tbdata['X']
            ypos = tbdata['Y']
            telapse = tbheader['ONTIME']
            obsidev = tbheader['OBS_ID']
            src_significance = tbdata['SRC_SIGNIFICANCE']
            netcts = tbdata['NET_COUNTS'] 
                        
            #Compute separation between catalogue position and 
            #real XMM position
            csrc = SkyCoord(RA,DEC,unit="deg",frame="icrs")
            ranew = csrc.ra.degree
            decnew = csrc.dec.degree       
            separation = culx.separation(csrc)
            sepdeg = separation.deg
            
            #Filter on counts
            ranew = ranew[netcts>=ctthreshwav]
            decnew = decnew[netcts>=ctthreshwav]
            xpos = xpos[netcts>=ctthreshwav]
            ypos = ypos[netcts>=ctthreshwav]
            sepdeg = sepdeg[netcts>=ctthreshwav]
            src_significance = src_significance[netcts>=ctthreshwav]
            netcts = netcts[netcts>=ctthreshwav]
            
            #Filter on separation
            ranewfilt = ranew[sepdeg<=septhresh]
            decnewfilt = decnew[sepdeg<=septhresh]
            xposfilt = xpos[sepdeg<=septhresh]
            yposfilt = ypos[sepdeg<=septhresh]
            netcts = netcts[sepdeg<=septhresh] 
            src_significance = src_significance[sepdeg<=septhresh]
            sepdegfilt = sepdeg[sepdeg<=septhresh]
            septhreshnew = septhresh
                    
            if(len(ranewfilt)==0):
                counter += 1
                counter2 += 1
            
            if(len(ranewfilt)==1):
                counter += 1
                
                #Create ellipse region filter (i.e. output of wavdetect) 
                regionfile = galaxies[k] + "/" + ObsId + "/proc/EPNimage" +\
                             str(obsindex) + "_reg.fits"   
                             
                with open(regionfile) as fi:
                    for line in fi:
                        inf = line.split()[0]
                        
                        XPOS = float(inf.split("ellipse")[1].split(",")[0][1:])
                        YPOS = float(inf.split("ellipse")[1].split(",")[1])
                        semimajor = float(inf.split("ellipse")[1].split(",")[2])
                        semiminor = float(inf.split("ellipse")[1].split(",")[3])
                        posangle = float(inf.split("ellipse")[1].split\
                                          (",")[4][0:-1])
                        
                        for m in range(len(xposfilt)):
                            
                            sep_physical =\
                                np.sqrt((xposfilt[m]-XPOS)**2 +\
                                        (yposfilt[m]-YPOS)**2)
                            sep_arcsec = pixscale*sep_physical
                            
                            if(sep_arcsec<=septhreshnew):
                                
                                expression = '#XMMEA_EP && (PATTERN<=4) && ' +\
                                              '(PI>=' + str(pimin) + ')' +\
                                              ' && ' +\
                                              '(PI<=' + str(pimax) + ')' +\
                                              ' && ' +\
                                              '(' + '(X,Y) in ' + str(inf) + ')'
                                
                                regarr = []
                                regarr.append(inf)
                                
                                regfile = "src_" + galaxies[k] + "_" +\
                                          ObsId + "_obs" + obsindex +\
                                          "_src" + nrepind + "_ref.reg"
    
                                # np.savetxt(regfile,regarr,fmt='%s')
                                
                                radet = ranewfilt[0]
                                decdet = decnewfilt[0]
                                
                                unfiltevfile = "epn" + obsindex +\
                                                ".fits"
                                # print(ObsId,unfiltevfile)

                                unfiltevfilecp = "epn" + obsindex +\
                                                  "cp.fits"
                                                 
                                filtevfile = "epn_obs" + obsindex +\
                                              "_src" + nrepind + "_ref.fits" 
                                                            
                                filtevfilebkg = "epn_bkg" + obsindex +\
                                                "_src" + nrepind + "_ref.fits"
                                                
                                filtlc = "epn_net" + obsindex +\
                                          "_src" + nrepind + "_ref.lc"
                                          
                                regionfile = "EPNimage" + (obsindex) +\
                                              "_reg.fits"
                                
                                #Update CCF and ODF files
                                loc = "/Users/raj1294/Documents/" +\
                                      "SEAWIND/Pulsation_search/new/" +\
                                      galaxies[k] + "/" + ObsId + "/proc/"
                                
                                comm1f = "cd " + loc
                                comm1fb = "export SAS_CCF=" + loc + "ccf.cif"
                                for sasfile in glob.glob(loc + "*.SAS"):
                                    comm1fc = "export SAS_ODF=" + sasfile
                                
                                comm1fd = "cp " + unfiltevfile +\
                                          " " + unfiltevfilecp
    
                                #Barycenter
                                comm1g = "barycen table=" + unfiltevfilecp +\
                                          ":EVENTS " +\
                                          "timecolumn=TIME" +\
                                          " withsrccoordinates=yes srcra=" +\
                                          str(radet) + " srcdec=" + str(decdet) +\
                                          " ephemeris=DE405"
                                
                                #Extract filtered event file
                                comm2f = "evselect table=" + unfiltevfilecp +\
                                          " expression='" + expression + "'" +\
                                          " withfilteredset=Y filteredset='" +\
                                          filtevfile + "'" 
    
                                #Extract filtered lightcurve
                                comm2g = "evselect table=" + unfiltevfilecp +\
                                          " expression='" + expression + "'" +\
                                          " rateset=Y rateset='" +\
                                          filtlc + "'" + " maketimecolumn=Y" +\
                                          " timebinsize=" +\
                                          str(bintime) + " makeratecolumn=Y"
                                
                                comm3f = "cd ../../../"
    
                                string4.append(comm1f)
                                string4.append(comm1fb)
                                string4.append(comm1fc)
                                string4.append(comm1fd)
                                string4.append(comm1g)
                                string4.append(comm2f)
                                string4.append(comm2g)
                                string4.append(comm3f)
                                string4.append("")
    
                                counter2 += 1
    
            if(len(ranewfilt)>1):
                
                while(len(ranewfilt)>1):
        
                    ranewfilt = ranewfilt[sepdegfilt<=septhreshnew]
                    decnewfilt = decnewfilt[sepdegfilt<=septhreshnew]
                    xposfilt = xposfilt[sepdegfilt<=septhreshnew]
                    yposfilt = yposfilt[sepdegfilt<=septhreshnew]
                    netcts = netcts[sepdegfilt<=septhreshnew] 
                    src_significance = src_significance[sepdegfilt<=septhreshnew]
                    sepdegfilt = sepdegfilt[sepdegfilt<=septhreshnew]
                    septhreshnew -= dsep
                    
                    if(septhreshnew<0):
                        break
                    if(len(ranewfilt)<=1):
                        counter += 1
                        
                        #Create ellipse region filter (i.e. output of wavdetect) 
                        regionfile = galaxies[k] + "/" + ObsId +\
                                      "/proc/EPNimage" +\
                                      str(obsindex) + "_reg.fits"
                                     
                        with open(regionfile) as fi:
                            for line in fi:
                                inf = line.split()[0]
                                XPOS = float(inf.split\
                                        ("ellipse")[1].split(",")[0][1:])
                                YPOS = float(inf.split\
                                        ("ellipse")[1].split(",")[1])
                                semimajor = float(inf.split\
                                        ("ellipse")[1].split(",")[2])
                                semiminor = float(inf.split\
                                        ("ellipse")[1].split(",")[3])
                                posangle = float(inf.split\
                                        ("ellipse")[1].split(",")[4][0:-1])
                                
                                for m2 in range(len(xposfilt)):
                                    
                                    sep_physical =\
                                        np.sqrt((xposfilt[m2]-XPOS)**2 +\
                                                (yposfilt[m2]-YPOS)**2)
                                    sep_arcsec = pixscale*sep_physical
                                    
                                    if(sep_arcsec<=septhreshnew):
                                            
                                        expression = '#XMMEA_EP &&' +\
                                        ' (PATTERN<=4) && ' +\
                                        '(PI>=' + str(pimin) + ')' +\
                                        ' && ' +\
                                        '(PI<=' + str(pimax) + ')' +\
                                        ' && ' +\
                                        '(' + '(X,Y) in ' + str(inf) + ')'
                                        
                                        regarr = []
                                        regarr.append(inf)
                                        
                                        regfile = "src_" + galaxies[k] + "_" +\
                                                  ObsId + "_obs" + obsindex +\
                                                  "_src" + nrepind + "_ref.reg"
                                        
                                        # np.savetxt(regfile,regarr,fmt='%s')
                                        
                                        radet = ranewfilt[0]
                                        decdet = decnewfilt[0]
                                        
                                        unfiltevfile = "epn" + obsindex +\
                                                        ".fits"
                                        # print(ObsId,unfiltevfile)
                                        
                                        unfiltevfilecp = "epn" + obsindex +\
                                                          "cp.fits"
                                                         
                                        filtevfile = "epn_obs" + obsindex +\
                                                      "_src" + nrepind +\
                                                      "_ref.fits" 
                                                                                
                                        filtevfilebkg = "epn_bkg" + obsindex +\
                                                        "_src" + nrepind +\
                                                        "_ref.fits"
                                                        
                                        filtlc = "epn_net" + obsindex +\
                                                  "_src" + nrepind + "_ref.lc"
                                        
                                        #Update CCF and ODF files
                                        loc = "/Users/raj1294/Documents/" +\
                                              "SEAWIND/Pulsation_search/new/" +\
                                              galaxies[k] + "/" + ObsId +\
                                              "/proc/"
                                        
                                        comm1f = "cd " + loc
                                        comm1fb = "export SAS_CCF=" + loc +\
                                                  "ccf.cif"
                                        for sasfile in glob.glob(loc + "*.SAS"):
                                            comm1fc = "export SAS_ODF=" + sasfile
                                        comm1fd = "cp " + unfiltevfile +\
                                                  " " + unfiltevfilecp
    
                                        #Barycenter
                                        comm1g = "barycen table=" +\
                                                  unfiltevfilecp +\
                                                  ":EVENTS " + "timecolumn=TIME" +\
                                                  " withsrccoordinates=yes" +\
                                                  " srcra=" + str(radet) +\
                                                  " srcdec=" + str(decdet) +\
                                                  " ephemeris=DE405"
                                        
                                        #Extract filtered event file
                                        comm2f = "evselect table=" +\
                                                  unfiltevfilecp +\
                                                  " expression='" + expression +\
                                                  "'" +\
                                                  " withfilteredset=Y" +\
                                                  " filteredset='" +\
                                                  filtevfile + "'"
                                                                                
                                        #Extract filtered lightcurve
                                        comm2g = "evselect table=" +\
                                        unfiltevfilecp +\
                                        " expression='" + expression +\
                                        "'" + " rateset=Y rateset='" +\
                                        filtlc + "'" +\
                                                  " maketimecolumn=Y" +\
                                                  " timebinsize=" +\
                                                  str(bintime) +\
                                                  " makeratecolumn=Y"
                                        
                                        comm3f = "cd ../../../"
                                        
                                        
                                        string4.append(comm1f)
                                        string4.append(comm1fb)
                                        string4.append(comm1fc)
                                        string4.append(comm1fd)
                                        string4.append(comm1g)
                                        string4.append(comm2f)
                                        string4.append(comm2g)
                                        string4.append(comm3f)
                                        string4.append("")
                                        
                                        counter2 += 1
                                        
                        break

np.savetxt("filter_lc.sh",string4,fmt='%s',delimiter='   ')
os.system("chmod u+x filter_lc.sh")

np.savetxt("remove_evcp.sh",string5,fmt='%s',delimiter='   ')
os.system("chmod u+x remove_evcp.sh")

Gunique = np.unique(galaxies)
Nseg = 0
string6,string7 = [[],[]]
for k2 in range(len(Gunique)):
    
    if(Gunique[k2]=='NGC1313'):
                
        Comm1 = "mkdir " + Gunique[k2] + "/lightcurves/"
        Comm2 = "mkdir " + Gunique[k2] + "/lightcurves/accelsearch/"
        Comm3 = "mkdir " + Gunique[k2] +\
                "/lightcurves/accelsearch/candidates/"
        Comm3b = "cp Obs1.inf " + Gunique[k2] + "/lightcurves/"
                
        string6.append(Comm1)
        string6.append(Comm2)
        string6.append(Comm3)
        string6.append(Comm3b)
        
        for flc in sorted(glob.glob(Gunique[k2]+"/0*/proc/*.lc")): 
        
            hdulc = fits.open(flc)
            
            #ObsID
            obsidlc = hdulc[0].header['OBS_ID']
            #Visit
            obsnum = flc.split("_net")[1].split("_src")[0]
            #Source
            srcnum = flc.split("_src")[1].split("_ref")
            
            flcnew = "epn_net" + str(obsnum) + "_" + str(srcnum) +\
                     "_ref_" + str(obsidlc) + ".lc" 
            Comm4 = "cp " + flc + " " + Gunique[k2] + "/lightcurves/" +\
                    flcnew       
                    
            string6.append(Comm4)
        
        for fev in sorted(glob.glob(Gunique[k2]+\
                                    "/0*/proc/epn*obs*ref.fits")): 
            
            hduev = fits.open(fev)
            #ObsId
            obsidev = hduev[1].header['OBS_ID']
            #Visit
            obsnum = fev.split("_obs")[1].split("_src")[0]
            #Source
            srcnum = fev.split("_src")[1].split("_ref")[0] 
            
            telapse = hduev[1].header['ONTIME']
            tstart = hduev[1].header['TSTART']
                        
            Comm0 = "rm -f " + Gunique[k2] + "/" + obsidev + "/proc/" +\
                      "epn*net*ref.fits"
            
            #Choose Nseg depending on exposure time
            if (telapse > 100*ks):
                Nseg = 4
            
            if (telapse > 80*ks and telapse <= 100*ks):
                Nseg = 3
            
            if (telapse > 60*ks and telapse <= 80*ks):
                Nseg = 2
            
            if (telapse > 30*ks and telapse <= 60*ks):
                Nseg = 1
    
            if (telapse <= 30*ks):
                Nseg = 1
            
            if(obsidev=='0803990401'):
                
                Nseg = 1
                
            if(obsidev=='0803990601'):
            
                Nseg = 4  

            dtelapse = telapse/Nseg
                        
            for k3 in range(Nseg):
                
                #Copy event file
                tbegin = tstart + k3*dtelapse
                tend = tstart + (k3+1)*dtelapse
                
                #Segment number
                nsegnum = str(k3+1)
                
                fevnew = "epn_obs" + str(obsnum) + "_src" + str(srcnum) +\
                         "_seg" + str(nsegnum) + "_" + str(obsidev) +\
                         ".fits"
                          
                Comm5a = "evselect table=" + str(fev) + " expression='" +\
                        "TIME IN [" + str(tbegin) + ":" + str(tend) + "]' " +\
                        "withfilteredset=Y filteredset=" +\
                        Gunique[k2] + "/" + obsidev + "/proc/" + str(fevnew)
                                
                string6.append(Comm5a)
                
                #Copy LC
                Comm5b = "cp " + Gunique[k2] + "/" +\
                          obsidev + "/proc/" + fevnew +\
                          " " + Gunique[k2] + "/lightcurves/accelsearch/"      
                string6.append(Comm5b)
        
        #Run acceleration search and z-search
        Comm5 = "cd " + Gunique[k2] + "/lightcurves/"
        Comm6 = "python ~/Documents/SEAWIND/code/lc_presto.py"
        Comm7 = "cd accelsearch/"
        Comm8 = "python ~/Documents/SEAWIND/code/accn_search.py"
        Comm8b = "python ~/Documents/SEAWIND/code/evfile_zsearch.py"
        Comm9 = "cd ../../../"
        
        string6.append(Comm5)
        string6.append(Comm6)
        string6.append(Comm7)
        string6.append(Comm8)
        string6.append(Comm8b)
        string6.append(Comm9)

np.savetxt("pulse_search.sh",string6,fmt='%s',delimiter='    ')

    
