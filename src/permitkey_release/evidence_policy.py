"""Keyword evidence policy used to screen category assignments from description text."""
import re
from .common import HEADS
def bit(name): return 1 << HEADS.index(name)
def normalize(text):
    return re.sub(r'\s+',' ',str(text or '').replace('\\r',' ').replace('\\n',' ').replace('\\t',' ')).strip()
def key(text):return re.sub(r'[^A-Z0-9]+',' ',normalize(text).upper()).strip()
PHRASES={
 'SOLAR_PV':['SOLAR PV','PHOTOVOLTAIC','PHOTOVOLTAIC SYSTEM','SOLAR PANELS','SOLAR ELECTRIC'],
 'EV_CHARGER':['EV CHARGER','EV CHARGING STATION','ELECTRIC VEHICLE CHARGER','ELECTRIC VEHICLE CHARGING STATION'],
 'BATTERY_STORAGE':['BATTERY STORAGE','ENERGY STORAGE SYSTEM','BATTERY ENERGY STORAGE'],
 'HEAT_PUMP':['HEAT PUMP','HEAT PUMP INSTALLATION','REPLACE HEAT PUMP'],
 'ELECTRICAL_PANEL_UPGRADE':['PANEL CHANGE','PANEL UPGRADE','ELECTRICAL PANEL UPGRADE','REPLACE PANEL','ELECTRICAL PANEL REPLACEMENT'],
 'ROOFING':['REROOF','RE ROOF','RE ROOFING','REROOFING','REROOF PERMIT','ROOFING','ROOFING RES','ROOFING COMM','ROOF REPLACEMENT','REPLACE ROOF','REROOF ONLY','BUILDING ROOFING','ROOF REPAIR'],
 'POOL_SPA':['SWIMMING POOL','POOL','SPA','POOL SPA','SWIMMING POOL SPA'],
 'DEMOLITION':['DEMOLITION','DEMOLITION PERMIT','BUILDING DEMOLITION','DEMOLISH BUILDING'],
 'WATER_HEATER':['WATER HEATER','WATER HEATER REPLACEMENT','REPLACE WATER HEATER','HOT WATER HEATER','WATER HEATER PERMIT','TANKLESS WATER HEATER'],
 'FIRE_PROTECTION':['FIRE SPRINKLER','FIRE SPRINKLERS','FIRE ALARM','FIRE PROTECTION','FIRE SUPPRESSION'],
 'WINDOW_DOOR':['WINDOW REPLACEMENT','REPLACE WINDOWS','WINDOWS DOORS','WINDOW DOOR','DOOR REPLACEMENT'],
 'FENCE_WALL':['FENCE','FENCES','FENCE PERMIT','RETAINING WALL','RETAINING WALLS'],
 'DECK_PATIO_PORCH':['DECK','DECK PERMIT','PATIO','PATIO COVER','PORCH','COVERED PATIO','DECK PATIO PORCH'],
 'DRIVEWAY_PAVING':['DRIVEWAY','DRIVEWAY PERMIT','PAVING','SIDEWALK','DRIVEWAY PAVING'],
 'GRADING_SITEWORK':['GRADING','GRADING PERMIT','SITEWORK','SITE WORK','EXCAVATION'],
 'GARAGE_ACCESSORY':['GARAGE','CARPORT','GARAGE CARPORTS','DETACHED GARAGE','ACCESSORY DWELLING UNIT'],
 'SEWER_UTILITY':['SEWER REPAIR','SEWER LATERAL','WATER SERVICE','WATER SERVICE REPLACEMENT','SEWER CONNECTION'],
 'GAS':['GAS LINE','GAS PIPING','GAS TEST','FUEL GAS','GAS METER REPLACEMENT'],
 'SIGN':['SIGN','SIGNS','SIGN PERMIT','WALL SIGN','MONUMENT SIGN'],
 'STRUCTURAL':['FOUNDATION REPAIR','FOUNDATION','STRUCTURAL REPAIR','STRUCTURAL ALTERATION','SEISMIC RETROFIT'],
 'NEW_CONSTRUCTION':['NEW CONSTRUCTION','NEW RESIDENCE','NEW BUILDING','NEW SINGLE FAMILY RESIDENCE','NEW SINGLE FAMILY DWELLING'],
 'ALTERATION_REMODEL':['TENANT IMPROVEMENT','TENANT IMPROVEMENTS','REMODEL','REMODELING','REMODELING REPAIRS','RENOVATION','ADDITIONS ALTERATION','ADDITION ALTERATION','ALTERATIONS','ALTERATION','BASEMENT FINISH'],
 'ELECTRICAL':['ELECTRICAL','ELECTRICAL EL','ELECTRICAL PERMIT','GENERAL WIRING','RES ELECTRICAL EXISTING','ELECTRICAL RESIDENTIAL','ELECTRICAL COMMERCIAL'],
 'PLUMBING':['PLUMBING','PLUMBING PL','PLUMBING PERMIT','RES PLUMBING','INSIDE PLUMBING','OUTSIDE PLUMBING','REPIPE'],
 'MECHANICAL_HVAC':['HVAC','MECHANICAL','MECHANICAL ME','MECHANICAL PERMIT','FURNACE REPLACEMENT','REPLACE FURNACE','INSTALL FURNACE','INSTALL AC','INSTALL A C','AIR CONDITIONER','AIR CONDITIONING','REPLACE AIR CONDITIONER']
}
EXACT={key(t):bit(h) for h,values in PHRASES.items() for t in values}
PATTERNS=[
 r'\b(photovoltaic|solar\s+(pv|panel|electric))\b',
 r'\b(ev\s+charg\w*|electric\s+vehicle\s+charg\w*)\b',
 r'\b(battery\s+(storage|system)|energy\s+storage)\b',
 r'\bheat\s+pumps?\b',r'\b(panel\s+(upgrade|replacement|change)|replac\w*\s+(electrical\s+)?panel)\b',
 r'\b(re[ -]?roof\w*|roofing|roof\s+(repair|replacement)|replac\w*\s+roof)\b',
 r'\b(swimming\s+pool|pool|spa)\b',r'\b(demolish\w*|demolition)\b',
 r'\b(water\s+heater|tankless)\b',r'\b(fire\s+(alarm|sprinkler|suppression|protection))\b',
 r'\b(windows?|doors?|glazing)\b',r'\b(fences?|retaining\s+walls?)\b',
 r'\b(decks?|patios?|porch\w*)\b',r'\b(driveways?|paving|sidewalks?)\b',
 r'\b(grading|sitework|excavat\w*)\b',r'\b(garages?|carports?|accessory\s+dwelling)\b',
 r'\b(sewer|water\s+(service|main)|utility\s+main)\b',r'\b(gas\s+(line|piping|meter|test)|fuel\s+gas)\b',
 r'\b(signage|signs?|billboard)\b',r'\b(structural|foundation|seismic)\b',
 r'\bnew\s+(construction|building|residence|dwelling|single.family)\b',
 r'\b(remodel\w*|renovat\w*|alteration\w*|tenant\s+improvement\w*|basement\s+finish)\b',
 r'\b(electrical|wiring)\b',r'\b(plumbing|repipe\w*)\b',r'\b(hvac|mechanical|furnace|air\s+condition\w*)\b'
]
RX=[re.compile(x,re.I) for x in PATTERNS]
BLOCK=re.compile(r'\b(no|not|without|exclude\w*|cancel\w*|void\w*|withdraw\w*)\b',re.I)
ADDRESS=re.compile(r'^\d+\s+.{0,100}\b(street|st|avenue|ave|road|rd|drive|dr|lane|ln|blvd)\.?$',re.I)
def assess(text):
    t=normalize(text)
    if not t:return t,0,0,'empty'
    if BLOCK.search(t):return t,0,0,'negation_or_administrative_hold'
    if ADDRESS.fullmatch(t):return t,0,0,'address_only_hold'
    direct=EXACT.get(key(t),0)
    if direct:return t,direct,direct,'explicit_category_mapping'
    allowed=sum(1<<i for i,p in enumerate(RX) if p.search(t))
    return t,allowed,0,'model_candidate' if allowed else 'no_category_specific_evidence'
