// The constants of SciPy `scipy.constants`: the numbers of the module and
// the table of the physical constants. The table holds the values of
// CODATA 2022, as SciPy 1.18 holds them. The units hold the SI base
// units unless the name says otherwise.
// The file holds data only. The functions that read the table live in
// the array_constants module.
namespace constants;

const Avogadro: f64 = 6.02214076e+23;
const Boltzmann: f64 = 1.380649e-23;
const Btu: f64 = 1055.05585262;
const Btu_IT: f64 = 1055.05585262;
const Btu_th: f64 = 1054.3502644888888;
const G: f64 = 6.6743e-11;
const Julian_year: f64 = 31557600.0;
const N_A: f64 = 6.02214076e+23;
const Planck: f64 = 6.62607015e-34;
const R: f64 = 8.31446261815324;
const Rydberg: f64 = 10973731.568157;
const Stefan_Boltzmann: f64 = 5.6703744191844314e-08;
const Wien: f64 = 0.0028977719551851727;
const acre: f64 = 4046.8564223999992;
const alpha: f64 = 0.0072973525643;
const angstrom: f64 = 1e-10;
const arcmin: f64 = 0.0002908882086657216;
const arcminute: f64 = 0.0002908882086657216;
const arcsec: f64 = 4.84813681109536e-06;
const arcsecond: f64 = 4.84813681109536e-06;
const astronomical_unit: f64 = 149597870700.0;
const atm: f64 = 101325.0;
const atmosphere: f64 = 101325.0;
const atomic_mass: f64 = 1.66053906892e-27;
const atto: f64 = 1e-18;
const au: f64 = 149597870700.0;
const bar: f64 = 100000.0;
const barrel: f64 = 0.15898729492799998;
const bbl: f64 = 0.15898729492799998;
const blob: f64 = 175.12683524647636;
const c: f64 = 299792458.0;
const calorie: f64 = 4.184;
const calorie_IT: f64 = 4.1868;
const calorie_th: f64 = 4.184;
const carat: f64 = 0.0002;
const centi: f64 = 0.01;
const day: f64 = 86400.0;
const deci: f64 = 0.1;
const degree: f64 = 0.017453292519943295;
const degree_Fahrenheit: f64 = 0.5555555555555556;
const deka: f64 = 10.0;
const dyn: f64 = 1e-05;
const dyne: f64 = 1e-05;
const e: f64 = 1.602176634e-19;
const eV: f64 = 1.602176634e-19;
const electron_mass: f64 = 9.1093837139e-31;
const electron_volt: f64 = 1.602176634e-19;
const elementary_charge: f64 = 1.602176634e-19;
const epsilon_0: f64 = 8.8541878188e-12;
const erg: f64 = 1e-07;
const exa: f64 = 1e+18;
const exbi: f64 = 1.152921504606847e+18;
const femto: f64 = 1e-15;
const fermi: f64 = 1e-15;
const fine_structure: f64 = 0.0072973525643;
const fluid_ounce: f64 = 2.9573529562499998e-05;
const fluid_ounce_US: f64 = 2.9573529562499998e-05;
const fluid_ounce_imp: f64 = 2.84130625e-05;
const foot: f64 = 0.30479999999999996;
const g: f64 = 9.80665;
const gallon: f64 = 0.0037854117839999997;
const gallon_US: f64 = 0.0037854117839999997;
const gallon_imp: f64 = 0.00454609;
const gas_constant: f64 = 8.31446261815324;
const gibi: f64 = 1073741824.0;
const giga: f64 = 1000000000.0;
const golden: f64 = 1.618033988749895;
const golden_ratio: f64 = 1.618033988749895;
const grain: f64 = 6.479891e-05;
const gram: f64 = 0.001;
const gravitational_constant: f64 = 6.6743e-11;
const h: f64 = 6.62607015e-34;
const hbar: f64 = 1.0545718176461565e-34;
const hectare: f64 = 10000.0;
const hecto: f64 = 100.0;
const horsepower: f64 = 745.6998715822701;
const hour: f64 = 3600.0;
const hp: f64 = 745.6998715822701;
const inch: f64 = 0.0254;
const k: f64 = 1.380649e-23;
const kgf: f64 = 9.80665;
const kibi: f64 = 1024.0;
const kilo: f64 = 1000.0;
const kilogram_force: f64 = 9.80665;
const kmh: f64 = 0.2777777777777778;
const knot: f64 = 0.5144444444444445;
const lb: f64 = 0.45359236999999997;
const lbf: f64 = 4.4482216152605;
const light_year: f64 = 9460730472580800.0;
const liter: f64 = 0.001;
const litre: f64 = 0.001;
const long_ton: f64 = 1016.0469088;
const m_e: f64 = 9.1093837139e-31;
const m_n: f64 = 1.67492750056e-27;
const m_p: f64 = 1.67262192595e-27;
const m_u: f64 = 1.66053906892e-27;
const mach: f64 = 340.5;
const mebi: f64 = 1048576.0;
const mega: f64 = 1000000.0;
const metric_ton: f64 = 1000.0;
const micro: f64 = 1e-06;
const micron: f64 = 1e-06;
const mil: f64 = 2.5399999999999997e-05;
const mile: f64 = 1609.3439999999998;
const milli: f64 = 0.001;
const minute: f64 = 60.0;
const mmHg: f64 = 133.32236842105263;
const mph: f64 = 0.44703999999999994;
const mu_0: f64 = 1.25663706127e-06;
const nano: f64 = 1e-09;
const nautical_mile: f64 = 1852.0;
const neutron_mass: f64 = 1.67492750056e-27;
const ounce: f64 = 0.028349523124999998;
const oz: f64 = 0.028349523124999998;
const parsec: f64 = 3.085677581491367e+16;
const pebi: f64 = 1125899906842624.0;
const peta: f64 = 1000000000000000.0;
const pi: f64 = 3.141592653589793;
const pico: f64 = 1e-12;
const point: f64 = 0.00035277777777777776;
const pound: f64 = 0.45359236999999997;
const pound_force: f64 = 4.4482216152605;
const proton_mass: f64 = 1.67262192595e-27;
const psi: f64 = 6894.757293168361;
const pt: f64 = 0.00035277777777777776;
const quecto: f64 = 1e-30;
const quetta: f64 = 1e+30;
const ronna: f64 = 1e+27;
const ronto: f64 = 1e-27;
const short_ton: f64 = 907.1847399999999;
const sigma: f64 = 5.6703744191844314e-08;
const slinch: f64 = 175.12683524647636;
const slug: f64 = 14.593902937206364;
const speed_of_light: f64 = 299792458.0;
const speed_of_sound: f64 = 340.5;
const stone: f64 = 6.3502931799999995;
const survey_foot: f64 = 0.3048006096012192;
const survey_mile: f64 = 1609.3472186944373;
const tebi: f64 = 1099511627776.0;
const tera: f64 = 1000000000000.0;
const ton_TNT: f64 = 4184000000.0;
const torr: f64 = 133.32236842105263;
const troy_ounce: f64 = 0.031103476799999998;
const troy_pound: f64 = 0.37324172159999996;
const u: f64 = 1.66053906892e-27;
const week: f64 = 604800.0;
const yard: f64 = 0.9143999999999999;
const year: f64 = 31536000.0;
const yobi: f64 = 1.2089258196146292e+24;
const yocto: f64 = 1e-24;
const yotta: f64 = 1e+24;
const zebi: f64 = 1.1805916207174113e+21;
const zepto: f64 = 1e-21;
const zero_Celsius: f64 = 273.15;
const zetta: f64 = 1e+21;

// The names of the physical constants, one per line, in sorted order.
const NAMES: const u8[:] = "Angstrom star\nAvogadro constant\nBohr magneton\nBohr magneton in Hz/T\nBohr magneton in K/T\nBohr magneton in eV/T\nBohr magneton in inverse meter per tesla\nBohr magneton in inverse meters per tesla\nBohr radius\nBoltzmann constant\nBoltzmann constant in Hz/K\nBoltzmann constant in eV/K\nBoltzmann constant in inverse meter per kelvin\nBoltzmann constant in inverse meters per kelvin\nCompton wavelength\nCompton wavelength over 2 pi\nCopper x unit\nCu x unit\nFaraday constant\nFaraday constant for conventional electric current\nFermi coupling constant\nHartree energy\nHartree energy in eV\nJosephson constant\nLoschmidt constant (273.15 K, 100 kPa)\nLoschmidt constant (273.15 K, 101.325 kPa)\nMo x unit\nMolybdenum x unit\nNewtonian constant of gravitation\nNewtonian constant of gravitation over h-bar c\nPlanck constant\nPlanck constant in eV s\nPlanck constant in eV/Hz\nPlanck constant over 2 pi\nPlanck constant over 2 pi in eV s\nPlanck constant over 2 pi times c in MeV fm\nPlanck length\nPlanck mass\nPlanck mass energy equivalent in GeV\nPlanck temperature\nPlanck time\nRydberg constant\nRydberg constant times c in Hz\nRydberg constant times hc in J\nRydberg constant times hc in eV\nSackur-Tetrode constant (1 K, 100 kPa)\nSackur-Tetrode constant (1 K, 101.325 kPa)\nStefan-Boltzmann constant\nThomson cross section\nW to Z mass ratio\nWien displacement law constant\nWien frequency displacement law constant\nWien wavelength displacement law constant\nalpha particle mass\nalpha particle mass energy equivalent\nalpha particle mass energy equivalent in MeV\nalpha particle mass in u\nalpha particle molar mass\nalpha particle relative atomic mass\nalpha particle rms charge radius\nalpha particle-electron mass ratio\nalpha particle-proton mass ratio\natomic mass constant\natomic mass constant energy equivalent\natomic mass constant energy equivalent in MeV\natomic mass unit-electron volt relationship\natomic mass unit-hartree relationship\natomic mass unit-hertz relationship\natomic mass unit-inverse meter relationship\natomic mass unit-joule relationship\natomic mass unit-kelvin relationship\natomic mass unit-kilogram relationship\natomic unit of 1st hyperpolarizability\natomic unit of 1st hyperpolarizablity\natomic unit of 2nd hyperpolarizability\natomic unit of 2nd hyperpolarizablity\natomic unit of action\natomic unit of charge\natomic unit of charge density\natomic unit of current\natomic unit of electric dipole mom.\natomic unit of electric dipole moment\natomic unit of electric field\natomic unit of electric field gradient\natomic unit of electric polarizability\natomic unit of electric polarizablity\natomic unit of electric potential\natomic unit of electric quadrupole mom.\natomic unit of electric quadrupole moment\natomic unit of energy\natomic unit of force\natomic unit of length\natomic unit of mag. dipole mom.\natomic unit of mag. flux density\natomic unit of magn. dipole moment\natomic unit of magn. flux density\natomic unit of magnetizability\natomic unit of mass\natomic unit of mom.um\natomic unit of momentum\natomic unit of permittivity\natomic unit of time\natomic unit of velocity\ncharacteristic impedance of vacuum\nclassical electron radius\nconductance quantum\nconventional value of Josephson constant\nconventional value of ampere-90\nconventional value of coulomb-90\nconventional value of farad-90\nconventional value of henry-90\nconventional value of ohm-90\nconventional value of volt-90\nconventional value of von Klitzing constant\nconventional value of watt-90\ndeuteron g factor\ndeuteron mag. mom.\ndeuteron mag. mom. to Bohr magneton ratio\ndeuteron mag. mom. to nuclear magneton ratio\ndeuteron magn. moment\ndeuteron magn. moment to Bohr magneton ratio\ndeuteron magn. moment to nuclear magneton ratio\ndeuteron mass\ndeuteron mass energy equivalent\ndeuteron mass energy equivalent in MeV\ndeuteron mass in u\ndeuteron molar mass\ndeuteron relative atomic mass\ndeuteron rms charge radius\ndeuteron-electron mag. mom. ratio\ndeuteron-electron magn. moment ratio\ndeuteron-electron mass ratio\ndeuteron-neutron mag. mom. ratio\ndeuteron-neutron magn. moment ratio\ndeuteron-proton mag. mom. ratio\ndeuteron-proton magn. moment ratio\ndeuteron-proton mass ratio\nelectric constant\nelectron charge to mass quotient\nelectron g factor\nelectron gyromag. ratio\nelectron gyromag. ratio in MHz/T\nelectron gyromag. ratio over 2 pi\nelectron gyromagn. ratio\nelectron gyromagn. ratio over 2 pi\nelectron mag. mom.\nelectron mag. mom. anomaly\nelectron mag. mom. to Bohr magneton ratio\nelectron mag. mom. to nuclear magneton ratio\nelectron magn. moment\nelectron magn. moment anomaly\nelectron magn. moment to Bohr magneton ratio\nelectron magn. moment to nuclear magneton ratio\nelectron mass\nelectron mass energy equivalent\nelectron mass energy equivalent in MeV\nelectron mass in u\nelectron molar mass\nelectron relative atomic mass\nelectron to alpha particle mass ratio\nelectron to shielded helion mag. mom. ratio\nelectron to shielded helion magn. moment ratio\nelectron to shielded proton mag. mom. ratio\nelectron to shielded proton magn. moment ratio\nelectron volt\nelectron volt-atomic mass unit relationship\nelectron volt-hartree relationship\nelectron volt-hertz relationship\nelectron volt-inverse meter relationship\nelectron volt-joule relationship\nelectron volt-kelvin relationship\nelectron volt-kilogram relationship\nelectron-deuteron mag. mom. ratio\nelectron-deuteron magn. moment ratio\nelectron-deuteron mass ratio\nelectron-helion mass ratio\nelectron-muon mag. mom. ratio\nelectron-muon magn. moment ratio\nelectron-muon mass ratio\nelectron-neutron mag. mom. ratio\nelectron-neutron magn. moment ratio\nelectron-neutron mass ratio\nelectron-proton mag. mom. ratio\nelectron-proton magn. moment ratio\nelectron-proton mass ratio\nelectron-tau mass ratio\nelectron-triton mass ratio\nelementary charge\nelementary charge over h\nelementary charge over h-bar\nfine-structure constant\nfirst radiation constant\nfirst radiation constant for spectral radiance\nhartree-atomic mass unit relationship\nhartree-electron volt relationship\nhartree-hertz relationship\nhartree-inverse meter relationship\nhartree-joule relationship\nhartree-kelvin relationship\nhartree-kilogram relationship\nhelion g factor\nhelion mag. mom.\nhelion mag. mom. to Bohr magneton ratio\nhelion mag. mom. to nuclear magneton ratio\nhelion mass\nhelion mass energy equivalent\nhelion mass energy equivalent in MeV\nhelion mass in u\nhelion molar mass\nhelion relative atomic mass\nhelion shielding shift\nhelion-electron mass ratio\nhelion-proton mass ratio\nhertz-atomic mass unit relationship\nhertz-electron volt relationship\nhertz-hartree relationship\nhertz-inverse meter relationship\nhertz-joule relationship\nhertz-kelvin relationship\nhertz-kilogram relationship\nhyperfine transition frequency of Cs-133\ninverse fine-structure constant\ninverse meter-atomic mass unit relationship\ninverse meter-electron volt relationship\ninverse meter-hartree relationship\ninverse meter-hertz relationship\ninverse meter-joule relationship\ninverse meter-kelvin relationship\ninverse meter-kilogram relationship\ninverse of conductance quantum\njoule-atomic mass unit relationship\njoule-electron volt relationship\njoule-hartree relationship\njoule-hertz relationship\njoule-inverse meter relationship\njoule-kelvin relationship\njoule-kilogram relationship\nkelvin-atomic mass unit relationship\nkelvin-electron volt relationship\nkelvin-hartree relationship\nkelvin-hertz relationship\nkelvin-inverse meter relationship\nkelvin-joule relationship\nkelvin-kilogram relationship\nkilogram-atomic mass unit relationship\nkilogram-electron volt relationship\nkilogram-hartree relationship\nkilogram-hertz relationship\nkilogram-inverse meter relationship\nkilogram-joule relationship\nkilogram-kelvin relationship\nlattice parameter of silicon\nlattice spacing of ideal Si (220)\nlattice spacing of silicon\nluminous efficacy\nmag. constant\nmag. flux quantum\nmagn. constant\nmagn. flux quantum\nmolar Planck constant\nmolar Planck constant times c\nmolar gas constant\nmolar mass constant\nmolar mass of carbon-12\nmolar volume of ideal gas (273.15 K, 100 kPa)\nmolar volume of ideal gas (273.15 K, 101.325 kPa)\nmolar volume of silicon\nmuon Compton wavelength\nmuon Compton wavelength over 2 pi\nmuon g factor\nmuon mag. mom.\nmuon mag. mom. anomaly\nmuon mag. mom. to Bohr magneton ratio\nmuon mag. mom. to nuclear magneton ratio\nmuon magn. moment\nmuon magn. moment to Bohr magneton ratio\nmuon magn. moment to nuclear magneton ratio\nmuon mass\nmuon mass energy equivalent\nmuon mass energy equivalent in MeV\nmuon mass in u\nmuon molar mass\nmuon-electron mass ratio\nmuon-neutron mass ratio\nmuon-proton mag. mom. ratio\nmuon-proton magn. moment ratio\nmuon-proton mass ratio\nmuon-tau mass ratio\nnatural unit of action\nnatural unit of action in eV s\nnatural unit of energy\nnatural unit of energy in MeV\nnatural unit of length\nnatural unit of mass\nnatural unit of mom.um\nnatural unit of mom.um in MeV/c\nnatural unit of momentum\nnatural unit of momentum in MeV/c\nnatural unit of time\nnatural unit of velocity\nneutron Compton wavelength\nneutron Compton wavelength over 2 pi\nneutron g factor\nneutron gyromag. ratio\nneutron gyromag. ratio in MHz/T\nneutron gyromag. ratio over 2 pi\nneutron gyromagn. ratio\nneutron gyromagn. ratio over 2 pi\nneutron mag. mom.\nneutron mag. mom. to Bohr magneton ratio\nneutron mag. mom. to nuclear magneton ratio\nneutron magn. moment\nneutron magn. moment to Bohr magneton ratio\nneutron magn. moment to nuclear magneton ratio\nneutron mass\nneutron mass energy equivalent\nneutron mass energy equivalent in MeV\nneutron mass in u\nneutron molar mass\nneutron relative atomic mass\nneutron to shielded proton mag. mom. ratio\nneutron to shielded proton magn. moment ratio\nneutron-electron mag. mom. ratio\nneutron-electron magn. moment ratio\nneutron-electron mass ratio\nneutron-muon mass ratio\nneutron-proton mag. mom. ratio\nneutron-proton magn. moment ratio\nneutron-proton mass difference\nneutron-proton mass difference energy equivalent\nneutron-proton mass difference energy equivalent in MeV\nneutron-proton mass difference in u\nneutron-proton mass ratio\nneutron-tau mass ratio\nnuclear magneton\nnuclear magneton in K/T\nnuclear magneton in MHz/T\nnuclear magneton in eV/T\nnuclear magneton in inverse meter per tesla\nnuclear magneton in inverse meters per tesla\nproton Compton wavelength\nproton Compton wavelength over 2 pi\nproton charge to mass quotient\nproton g factor\nproton gyromag. ratio\nproton gyromag. ratio in MHz/T\nproton gyromag. ratio over 2 pi\nproton gyromagn. ratio\nproton gyromagn. ratio over 2 pi\nproton mag. mom.\nproton mag. mom. to Bohr magneton ratio\nproton mag. mom. to nuclear magneton ratio\nproton mag. shielding correction\nproton magn. moment\nproton magn. moment to Bohr magneton ratio\nproton magn. moment to nuclear magneton ratio\nproton magn. shielding correction\nproton mass\nproton mass energy equivalent\nproton mass energy equivalent in MeV\nproton mass in u\nproton molar mass\nproton relative atomic mass\nproton rms charge radius\nproton-electron mass ratio\nproton-muon mass ratio\nproton-neutron mag. mom. ratio\nproton-neutron magn. moment ratio\nproton-neutron mass ratio\nproton-tau mass ratio\nquantum of circulation\nquantum of circulation times 2\nreduced Compton wavelength\nreduced Planck constant\nreduced Planck constant in eV s\nreduced Planck constant times c in MeV fm\nreduced muon Compton wavelength\nreduced neutron Compton wavelength\nreduced proton Compton wavelength\nreduced tau Compton wavelength\nsecond radiation constant\nshielded helion gyromag. ratio\nshielded helion gyromag. ratio in MHz/T\nshielded helion gyromag. ratio over 2 pi\nshielded helion gyromagn. ratio\nshielded helion gyromagn. ratio over 2 pi\nshielded helion mag. mom.\nshielded helion mag. mom. to Bohr magneton ratio\nshielded helion mag. mom. to nuclear magneton ratio\nshielded helion magn. moment\nshielded helion magn. moment to Bohr magneton ratio\nshielded helion magn. moment to nuclear magneton ratio\nshielded helion to proton mag. mom. ratio\nshielded helion to proton magn. moment ratio\nshielded helion to shielded proton mag. mom. ratio\nshielded helion to shielded proton magn. moment ratio\nshielded proton gyromag. ratio\nshielded proton gyromag. ratio in MHz/T\nshielded proton gyromag. ratio over 2 pi\nshielded proton mag. mom.\nshielded proton mag. mom. to Bohr magneton ratio\nshielded proton mag. mom. to nuclear magneton ratio\nshielded proton magn. moment\nshielded proton magn. moment to Bohr magneton ratio\nshielded proton magn. moment to nuclear magneton ratio\nshielding difference of d and p in HD\nshielding difference of t and p in HT\nspeed of light in vacuum\nstandard acceleration of gravity\nstandard atmosphere\nstandard-state pressure\ntau Compton wavelength\ntau Compton wavelength over 2 pi\ntau energy equivalent\ntau mass\ntau mass energy equivalent\ntau mass energy equivalent in MeV\ntau mass in u\ntau molar mass\ntau-electron mass ratio\ntau-muon mass ratio\ntau-neutron mass ratio\ntau-proton mass ratio\ntriton g factor\ntriton mag. mom.\ntriton mag. mom. to Bohr magneton ratio\ntriton mag. mom. to nuclear magneton ratio\ntriton mass\ntriton mass energy equivalent\ntriton mass energy equivalent in MeV\ntriton mass in u\ntriton molar mass\ntriton relative atomic mass\ntriton to proton mag. mom. ratio\ntriton-electron mag. mom. ratio\ntriton-electron mass ratio\ntriton-neutron mag. mom. ratio\ntriton-proton mag. mom. ratio\ntriton-proton mass ratio\nunified atomic mass unit\nvacuum electric permittivity\nvacuum mag. permeability\nvon Klitzing constant\nweak mixing angle\n{220} lattice spacing of silicon";

// The units of the physical constants, one per line, in the order of the names.
const UNITS: const u8[:] = "m\nmol^-1\nJ T^-1\nHz T^-1\nK T^-1\neV T^-1\nm^-1 T^-1\nm^-1 T^-1\nm\nJ K^-1\nHz K^-1\neV K^-1\nm^-1 K^-1\nm^-1 K^-1\nm\nm\nm\nm\nC mol^-1\nC_90 mol^-1\nGeV^-2\nJ\neV\nHz V^-1\nm^-3\nm^-3\nm\nm\nm^3 kg^-1 s^-2\n(GeV/c^2)^-2\nJ Hz^-1\neV s\neV Hz^-1\nJ s\neV s\nMeV fm\nm\nkg\nGeV\nK\ns\nm^-1\nHz\nJ\neV\n\n\nW m^-2 K^-4\nm^2\n\nm K\nHz K^-1\nm K\nkg\nJ\nMeV\nu\nkg mol^-1\n\nm\n\n\nkg\nJ\nMeV\neV\nE_h\nHz\nm^-1\nJ\nK\nkg\nC^3 m^3 J^-2\nC^3 m^3 J^-2\nC^4 m^4 J^-3\nC^4 m^4 J^-3\nJ s\nC\nC m^-3\nA\nC m\nC m\nV m^-1\nV m^-2\nC^2 m^2 J^-1\nC^2 m^2 J^-1\nV\nC m^2\nC m^2\nJ\nN\nm\nJ T^-1\nT\nJ T^-1\nT\nJ T^-2\nkg\nkg m s^-1\nkg m s^-1\nF m^-1\ns\nm s^-1\nohm\nm\nS\nHz V^-1\nA\nC\nF\nH\nohm\nV\nohm\nW\n\nJ T^-1\n\n\nJ T^-1\n\n\nkg\nJ\nMeV\nu\nkg mol^-1\n\nm\n\n\n\n\n\n\n\n\nF m^-1\nC kg^-1\n\ns^-1 T^-1\nMHz T^-1\nMHz T^-1\ns^-1 T^-1\nMHz T^-1\nJ T^-1\n\n\n\nJ T^-1\n\n\n\nkg\nJ\nMeV\nu\nkg mol^-1\n\n\n\n\n\n\nJ\nu\nE_h\nHz\nm^-1\nJ\nK\nkg\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nC\nA J^-1\nA J^-1\n\nW m^2\nW m^2 sr^-1\nu\neV\nHz\nm^-1\nJ\nK\nkg\n\nJ T^-1\n\n\nkg\nJ\nMeV\nu\nkg mol^-1\n\n\n\n\nu\neV\nE_h\nm^-1\nJ\nK\nkg\nHz\n\nu\neV\nE_h\nHz\nJ\nK\nkg\nohm\nu\neV\nE_h\nHz\nm^-1\nK\nkg\nu\neV\nE_h\nHz\nm^-1\nJ\nkg\nu\neV\nE_h\nHz\nm^-1\nJ\nK\nm\nm\nm\nlm W^-1\nN A^-2\nWb\nN A^-2\nWb\nJ Hz^-1 mol^-1\nJ m mol^-1\nJ mol^-1 K^-1\nkg mol^-1\nkg mol^-1\nm^3 mol^-1\nm^3 mol^-1\nm^3 mol^-1\nm\nm\n\nJ T^-1\n\n\n\nJ T^-1\n\n\nkg\nJ\nMeV\nu\nkg mol^-1\n\n\n\n\n\n\nJ s\neV s\nJ\nMeV\nm\nkg\nkg m s^-1\nMeV/c\nkg m s^-1\nMeV/c\ns\nm s^-1\nm\nm\n\ns^-1 T^-1\nMHz T^-1\nMHz T^-1\ns^-1 T^-1\nMHz T^-1\nJ T^-1\n\n\nJ T^-1\n\n\nkg\nJ\nMeV\nu\nkg mol^-1\n\n\n\n\n\n\n\n\n\nkg\nJ\nMeV\nu\n\n\nJ T^-1\nK T^-1\nMHz T^-1\neV T^-1\nm^-1 T^-1\nm^-1 T^-1\nm\nm\nC kg^-1\n\ns^-1 T^-1\nMHz T^-1\nMHz T^-1\ns^-1 T^-1\nMHz T^-1\nJ T^-1\n\n\n\nJ T^-1\n\n\n\nkg\nJ\nMeV\nu\nkg mol^-1\n\nm\n\n\n\n\n\n\nm^2 s^-1\nm^2 s^-1\nm\nJ s\neV s\nMeV fm\nm\nm\nm\nm\nm K\ns^-1 T^-1\nMHz T^-1\nMHz T^-1\ns^-1 T^-1\nMHz T^-1\nJ T^-1\n\n\nJ T^-1\n\n\n\n\n\n\ns^-1 T^-1\nMHz T^-1\nMHz T^-1\nJ T^-1\n\n\nJ T^-1\n\n\n\n\nm s^-1\nm s^-2\nPa\nPa\nm\nm\nMeV\nkg\nJ\nMeV\nu\nkg mol^-1\n\n\n\n\n\nJ T^-1\n\n\nkg\nJ\nMeV\nu\nkg mol^-1\n\n\n\n\n\n\n\nkg\nF m^-1\nN A^-2\nohm\n\nm";

const COUNT: i64 = 445;

const VALUES: f64[445] = [
    1.00001495e-10, 6.02214076e+23, 9.2740100657e-24, 13996244917.1,
    0.67171381472, 5.7883817982e-05, 46.686447719, 46.68644814,
    5.29177210544e-11, 1.380649e-23, 20836619123.327576, 8.617333262145179e-05,
    69.50348004861274, 69.503457, 2.42631023538e-12, 3.8615926764e-13,
    1.00207697e-13, 1.00207697e-13, 96485.33212331001, 96485.3251,
    1.1663787e-05, 4.359744722206e-18, 27.211386245981, 483597848416983.6,
    2.6516458048837345e+25, 2.686780111798444e+25, 1.00209952e-13, 1.00209952e-13,
    6.6743e-11, 6.70883e-39, 6.62607015e-34, 4.135667662e-15,
    4.135667696923859e-15, 1.0545718e-34, 6.582119514e-16, 197.3269788,
    1.616255e-35, 2.176434e-08, 1.22089e+19, 1.416784e+32,
    5.391247e-44, 10973731.568157, 3289841960250000.0, 2.179872361103e-18,
    13.60569312299, -1.15170753496, -1.16487052149, 5.6703744191844314e-08,
    6.6524587051e-29, 0.88145, 0.0028977685, 58789257576.468254,
    0.0028977719551851727, 6.644657345e-27, 5.9719201997e-10, 3727.3794118,
    4.001506179129, 0.0040015061833, 4.001506179129, 1.6785e-15,
    7294.29954171, 3.972599690252, 1.66053906892e-27, 1.49241808768e-10,
    931.49410372, 931494103.72, 34231776.922, 2.25234272185e+23,
    751300662090000.0, 1.49241808768e-10, 10809540206700.0, 1.66053906892e-27,
    3.2063612996e-53, 3.20636151e-53, 6.2353799735e-65, 6.2353808e-65,
    1.0545718176461565e-34, 1.602176634e-19, 1081202386770.0, 0.0066236182375082,
    8.4783536198e-30, 8.47835309e-30, 514220675112.0, 9.7173624424e+21,
    1.64877727212e-41, 1.648777274e-41, 27.211386245981, 4.4865515185e-40,
    4.48655124e-40, 4.359744722206e-18, 8.2387235038e-08, 5.29177210544e-11,
    1.85480201315e-23, 235051.757077, 1.8548019e-23, 235051.757077,
    7.8910365794e-29, 9.1093837139e-31, 1.992851882e-24, 1.99285191545e-24,
    1.1126500562e-10, 2.4188843265864e-17, 2187691.26216, 376.730313412,
    2.8179403205e-15, 7.748091729863649e-05, 483597900000000.0, 1.0000000888714378,
    1.0000000888714378, 0.9999999822063325, 1.0000000177936679, 1.0000000177936679,
    1.0000001066651072, 25812.807, 1.0000001955365543, 0.8574382335,
    4.330735087e-27, 0.0004669754568, 0.8574382335, 4.33073482e-27,
    0.0004669754567, 0.8574382329, 3.3435837768e-27, 3.00506323491e-10,
    1875.612945, 2.013553212544, 0.00201355321466, 2.013553212544,
    2.12778e-15, -0.000466434555, -0.0004664345548, 3670.482967655,
    -0.44820652, -0.44820652, 0.3070122093, 0.3070122084,
    1.9990075012699, 8.8541878188e-12, -175882000838.0, -2.00231930436092,
    176085962784.0, 28024.9513861, 28024.95164, 176085962784.0,
    28024.9532, -9.2847646917e-24, 0.00115965218046, -1.00115965218046,
    -1838.281971877, -9.28476412e-24, 0.0011596521859, -1.0011596521859,
    -1838.28197107, 9.1093837139e-31, 8.187105788e-14, 0.51099895069,
    0.0005485799090441, 5.4857990962e-07, 0.0005485799090441, 0.0001370933554733,
    864.05823986, 864.058255, -658.2275856, -658.2275956,
    1.602176634e-19, 1.07354410083e-09, 0.036749322175665, 241798924208491.8,
    806554.3937349211, 1.602176634e-19, 11604.518121550082, 1.7826619216278975e-36,
    -2143.9234921, -2143.923493, 0.0002724437107629, 0.0001819543074649,
    206.7669881, 206.7669894, 0.0048363317, 960.92048,
    960.9205, 0.00054386734416, -658.21068789, -658.2106862,
    0.0005446170214889, 0.000287585, 0.0001819200062327, 1.602176634e-19,
    241798926200000.0, 1519267447878626.0, 0.0072973525643, 3.7417718521927573e-16,
    1.1910429723971884e-16, 2.92126231797e-08, 27.211386245981, 6579683920499900.0,
    21947463.136314, 4.359744722206e-18, 315775.02480398, 4.8508702095419e-35,
    -4.2552506995, -1.07461755198e-26, -0.00115874098083, -2.1276253498,
    5.0064127862e-27, 4.4995394185e-10, 2808.39161112, 3.014932246932,
    0.0030149322501, 3.014932246932, 5.9967029e-05, 5495.88527984,
    2.993152671552, 4.439821659e-24, 4.135667696923859e-15, 1.5198298460574e-16,
    3.3356409519815204e-09, 6.62607015e-34, 4.799243073366221e-11, 7.372497323812708e-51,
    9192631770.0, 137.035999177, 1.33102504824e-15, 1.2398419843320026e-06,
    4.5563352529132e-08, 299792458.0, 1.9864458571489286e-25, 0.014387768775039337,
    2.2102190943042335e-42, 12906.403729652257, 6700535247.1, 6.241509074460763e+18,
    2.2937122783969e+17, 1.5091901796421518e+33, 5.03411656754271e+24, 7.24297051603992e+22,
    1.1126500560536185e-17, 9.2510872884e-14, 8.617333262145179e-05, 3.1668115634564e-06,
    20836619123.327576, 69.50348004861274, 1.380649e-23, 1.5361791872403723e-40,
    6.0221407537e+26, 5.609588603804452e+35, 2.0614857887415e+34, 1.3563924896521321e+50,
    4.524438335443823e+41, 8.987551787368176e+16, 6.509657260728958e+39, 5.431020511e-10,
    1.920155716e-10, 1.920155762e-10, 683.0, 1.25663706127e-06,
    2.0678338484619295e-15, 1.2566370614359173e-06, 2.0678338484619295e-15, 3.990312712893431e-10,
    0.119626565582, 8.31446261815324, 0.00100000000105, 0.0120000000126,
    0.02271095464148557, 0.022413969545014137, 1.205883199e-05, 1.17344411e-14,
    1.867594308e-15, -2.00233184123, -4.4904483e-26, 0.00116592062,
    -0.00484197048, -8.89059704, -4.49044799e-26, -0.00484197045,
    -8.89059698, 1.883531627e-28, 1.692833804e-11, 105.6583755,
    0.1134289257, 0.0001134289258, 206.7682827, 0.1124545168,
    -3.183345146, -3.183345118, 0.1126095262, 0.0594635,
    1.0545718176461565e-34, 6.582119569509067e-16, 8.187105788e-14, 0.51099895069,
    3.8615926744e-13, 9.1093837139e-31, 2.730924488e-22, 0.5109989461,
    2.730924488e-22, 0.5109989461, 1.28808866644e-21, 299792458.0,
    1.31959090382e-15, 2.1001941536e-16, -3.82608552, 183247174.0,
    29.1646935, 29.1646933, 183247174.0, 29.164695,
    -9.6623653e-27, -0.00104187565, -1.91304276, -9.6623645e-27,
    -0.00104187563, -1.91304273, 1.67492750056e-27, 1.50534976514e-10,
    939.56542194, 1.00866491606, 0.00100866491712, 1.00866491606,
    -0.68499694, -0.68499694, 0.00104066884, 0.00104066882,
    1838.683662, 8.89248408, -0.68497935, -0.68497934,
    2.30557461e-30, 2.07214712e-13, 1.29333251, 0.00138844948,
    1.00137841946, 0.528779, 5.0507837393e-27, 0.00036582677706,
    7.6225932188, 3.15245125417e-08, 0.0254262341009, 0.02542623432,
    1.3214098536e-15, 2.10308910109e-16, 95788331.43, 5.5856946893,
    267522187.08, 42.577478461, 42.57747892, 267522187.08,
    42.5774813, 1.41060679545e-26, 0.0015210322023, 2.79284734463,
    2.56715e-05, 1.41060671e-26, 0.001521032206, 2.792847351,
    2.56715e-05, 1.67262192595e-27, 1.50327761802e-10, 938.27208943,
    1.0072764665789, 0.00100727646764, 1.0072764665789, 8.4075e-16,
    1836.152673426, 8.88024338, -1.45989802, -1.45989805,
    0.99862347797, 0.528051, 0.00036369475467, 0.00072738950934,
    3.8615926744e-13, 1.0545718176461565e-34, 6.582119569509067e-16, 197.3269804593025,
    1.867594306e-15, 2.100194152e-16, 2.10308910051e-16, 1.110538e-16,
    0.014387768775039337, 203789460.78, 32.434100033, 32.43409966,
    203789460.78, 32.4341015, -1.07455311035e-26, -0.00115867149457,
    -2.1274977624, -1.074553024e-26, -0.001158671474, -2.127497723,
    -0.76176657721, -0.761766562, -0.7617861334, -0.7617861313,
    267515319.4, 42.57638543, 42.57638507, 1.410570583e-26,
    0.0015209931551, 2.792775648, 1.41057047e-26, 0.001520993132,
    2.792775604, 1.9877e-08, 2.3945e-08, 299792458.0,
    9.80665, 101325.0, 100000.0, 6.97771e-16,
    1.11056e-16, 1776.86, 3.16754e-27, 2.84684e-10,
    1776.82, 1.90754, 0.00190754, 3477.23,
    16.817, 1.89115, 1.89376, 5.95792493,
    1.5046095178e-26, 0.0016223936648, 2.978962465, 5.0073567512e-27,
    4.5003878119e-10, 2808.92113668, 3.01550071597, 0.00301550071913,
    3.01550071597, 1.0666399189, -0.001620514423, 5496.92153551,
    -1.55718553, 1.066639908, 2.99371703403, 1.66053906892e-27,
    8.8541878188e-12, 1.25663706127e-06, 25812.807459304513, 0.22305,
    1.920155714e-10,
];

const UNCERTAINTIES: f64[445] = [
    9e-17, 0.0, 2.9e-33, 4.4,
    2.1e-10, 1.8e-14, 1.5e-08, 2.9e-07,
    8.2e-21, 0.0, 0.0, 0.0,
    0.0, 4e-05, 7.6e-22, 1.8e-22,
    2.8e-20, 2.8e-20, 0.0, 0.0012,
    6e-12, 4.8e-30, 3e-11, 0.0,
    0.0, 0.0, 5.3e-20, 5.3e-20,
    1.5e-15, 1.5e-43, 0.0, 2.5e-23,
    0.0, 1.3e-42, 4e-24, 1.2e-06,
    1.8e-40, 2.4e-13, 140000000000000.0, 1.6e+27,
    6e-49, 1.2e-05, 3600.0, 2.4e-30,
    1.5e-11, 4.7e-10, 4.7e-10, 0.0,
    6.2e-38, 0.00013, 5.1e-09, 0.0,
    0.0, 2.1e-36, 1.9e-19, 1.2e-06,
    6.2e-11, 1.2e-12, 6.2e-11, 2.1e-18,
    1.7e-07, 7e-11, 5.2e-37, 4.6e-20,
    2.9e-07, 0.29, 0.011, 70000000000000.0,
    230000.0, 4.6e-20, 3400.0, 5.2e-37,
    1.5e-62, 2.8e-60, 3.9e-74, 1.1e-71,
    0.0, 0.0, 510.0, 7.2e-15,
    1.3e-39, 7.3e-37, 80.0, 3000000000000.0,
    5.1e-51, 1.6e-49, 3e-11, 1.4e-49,
    3.9e-47, 4.8e-30, 1.3e-17, 8.2e-21,
    5.8e-33, 7.3e-05, 1.6e-30, 7.3e-05,
    4.9e-38, 2.8e-40, 2.4e-32, 3.1e-34,
    1.7e-20, 2.6e-29, 0.00034, 5.9e-08,
    1.3e-24, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 2.2e-09,
    1.1e-35, 1.2e-12, 2.2e-09, 3.8e-34,
    5e-12, 9.2e-09, 1e-36, 9.4e-20,
    5.8e-07, 1.5e-11, 6.3e-13, 1.5e-11,
    2.7e-19, 1.2e-12, 5e-12, 6.3e-08,
    1.1e-07, 1.1e-07, 7.9e-10, 4.5e-09,
    8.4e-12, 1.4e-21, 55.0, 3.6e-13,
    55.0, 8.7e-06, 0.00017, 55.0,
    0.0024, 2.9e-33, 1.8e-13, 1.8e-13,
    3.2e-08, 8e-31, 3.8e-12, 3.8e-12,
    8.5e-07, 2.8e-40, 2.6e-23, 1.6e-10,
    9.7e-15, 1.7e-16, 9.7e-15, 3.2e-15,
    7e-07, 1e-05, 2.7e-06, 7.1e-06,
    0.0, 3.3e-19, 4e-14, 0.0,
    0.0, 0.0, 0.0, 0.0,
    5.6e-06, 2.3e-05, 4.7e-15, 5.3e-15,
    4.6e-06, 5.4e-06, 1.1e-10, 0.00023,
    0.00023, 2.2e-13, 1.9e-07, 6.6e-06,
    9.4e-15, 1.9e-08, 6.8e-15, 0.0,
    1500000.0, 0.0, 1.1e-12, 0.0,
    0.0, 9.1e-18, 3e-11, 7200.0,
    2.4e-05, 4.8e-30, 3.4e-07, 5.3e-47,
    3.4e-09, 9.3e-36, 9.4e-13, 1.7e-09,
    1.6e-36, 1.4e-19, 8.8e-07, 7.4e-11,
    9.4e-13, 7.4e-11, 2.3e-11, 1.6e-07,
    7e-11, 1.4e-33, 0.0, 1.7e-28,
    0.0, 0.0, 0.0, 0.0,
    0.0, 2.1e-08, 4.1e-25, 0.0,
    5e-20, 0.0, 0.0, 0.0,
    0.0, 0.0, 2.1, 0.0,
    250000.0, 0.0, 0.0, 0.0,
    0.0, 2.9e-23, 0.0, 3.5e-18,
    0.0, 0.0, 0.0, 0.0,
    1.9e+17, 0.0, 2.2e+22, 0.0,
    0.0, 0.0, 0.0, 8.9e-18,
    3.2e-18, 5e-18, 0.0, 2e-16,
    0.0, 0.0, 0.0, 0.0,
    5.4e-11, 0.0, 3.1e-13, 3.7e-12,
    0.0, 0.0, 6e-13, 2.6e-22,
    4.2e-23, 8.2e-10, 1e-33, 4.1e-10,
    1.1e-10, 2e-07, 4e-33, 1.3e-10,
    2.3e-07, 4.2e-36, 3.8e-19, 2.3e-06,
    2.5e-09, 2.5e-12, 4.6e-06, 2.5e-09,
    7.1e-08, 8.9e-08, 2.5e-09, 4e-06,
    0.0, 0.0, 2.6e-23, 1.6e-10,
    1.2e-22, 2.8e-40, 3.4e-30, 3.1e-09,
    3.4e-30, 3.1e-09, 4e-31, 0.0,
    6.7e-25, 1.4e-25, 9e-07, 43.0,
    6.9e-06, 6.9e-06, 43.0, 7.3e-06,
    2.3e-33, 2.5e-10, 4.5e-07, 2.4e-33,
    2.5e-10, 4.5e-07, 8.5e-37, 7.6e-20,
    4.8e-07, 4e-10, 5.1e-13, 4e-10,
    1.6e-07, 1.6e-07, 2.4e-10, 2.5e-10,
    7.4e-07, 2e-07, 1.6e-07, 1.6e-07,
    6.7e-37, 6e-20, 3.8e-07, 4e-10,
    4e-10, 3.6e-05, 1.6e-36, 1.1e-13,
    2.4e-09, 9.8e-18, 7.9e-12, 1.6e-10,
    4.1e-25, 9.7e-26, 0.03, 1.6e-09,
    0.11, 1.8e-08, 2.9e-07, 0.11,
    3.7e-06, 6e-36, 4.5e-13, 8.2e-10,
    4.1e-09, 1.2e-33, 1.5e-11, 2.8e-08,
    4.1e-09, 5.2e-37, 4.7e-20, 2.9e-07,
    8.3e-12, 3.1e-13, 8.3e-12, 6.4e-19,
    3.2e-08, 2e-07, 3.4e-07, 3.4e-07,
    4e-10, 3.6e-05, 1.1e-13, 2.3e-13,
    1.2e-22, 0.0, 0.0, 0.0,
    4.2e-23, 1.1e-25, 6.6e-26, 7.5e-21,
    0.0, 0.18, 2.8e-08, 4.3e-07,
    0.18, 2.8e-06, 9.3e-36, 9.4e-13,
    1.7e-09, 9.3e-34, 1.4e-11, 2.5e-08,
    6.6e-10, 1.2e-08, 3.1e-09, 3.3e-09,
    1.1, 1.7e-07, 5.3e-07, 5.8e-35,
    6.2e-12, 1.1e-08, 1.2e-33, 1.6e-11,
    3e-08, 1e-12, 2e-12, 0.0,
    0.0, 0.0, 0.0, 4.7e-20,
    1e-20, 0.12, 2.1e-31, 1.9e-14,
    0.16, 0.00013, 1.3e-07, 0.23,
    0.0011, 0.00013, 0.00013, 1.2e-08,
    3e-35, 3.2e-12, 5.9e-09, 1.6e-36,
    1.4e-19, 8.8e-07, 1e-10, 9.4e-13,
    1e-10, 2.1e-09, 2.1e-11, 2.1e-07,
    3.7e-07, 1e-08, 1e-10, 5.2e-37,
    1.4e-21, 2e-16, 0.0, 0.00023,
    3.2e-18,
];
