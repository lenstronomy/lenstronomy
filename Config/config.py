__author__ = 'sibirrer'



# ==========================================================
# S Y S T E M  S P E C I F I C A T I O N S
# ==========================================================

system_name = 'RXJ1131-1231' #(string); accepts system names in the "database"
data_file = 'j8oi74010_drz.fits'#'RXJ1131_1231_test.fits'    #(str); name of the data file to be considered
#data_file = 'j8oi74010_drz.fits'    #(str); name of the data file to be considered


# ==========================================================
# C O S M O L O G Y  S P E C I F I C A T I O N S
# ==========================================================

cosmo_file = 'PyCosmo_param.ini'    #(string); parameter file of cosmological parameters, must be in the folder /Config



# ==========================================================
# L E N S  M O D E L  O P T I O N S
# ==========================================================

lens_type = 'SPEP'          #(string); accepts 'SIS', 'GAUSSIAN', 'SPEP', can be extended



# ==========================================================
# S O U R C E  M O D E L  O P T I O N S
# ==========================================================

source_type = 'GAUSSIAN'    #(string); accepts 'GAUSSIAN', can be extended



# ==========================================================
# I M A G E  M A K I N G  C O N F I G U R A T I O N S
# ==========================================================

subgrid_res = 10            #(int); factor of subpixel resolution for surface brightness calculation
numPix = 50                #number of pixels in raw/column of the cutout image
psf_type = 'GAUSSIAN'       #(str); accepts 'GAUSSIAN'



# ==========================================================
# M C M C  C O N F I G
# ==========================================================

num_burn = 100              #(int); number of burn in sampling
num_sample = 100            #(int); number of sampling



# ==========================================================
# X 2  O P T I O N S
# ==========================================================

x2_simple = True           #(bool); if true, choses simple X2 form



# ==========================================================
# W O R K F L O W  O P T I O N S
# ==========================================================

run_mcmc = True             #if True, run mcmc chain
run_pso = False              #if True, run PSO chain