import cdsapi
import sys

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type':'reanalysis',
        'format':'grib',
        'variable':['100m_u_component_of_wind', '100m_v_component_of_wind'],
        'year': sys.argv[1],
        'month': sys.argv[2],
        'day': sys.argv[3],
        'area': sys.argv[4],
        'time':['00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00'
            ],
    },
    sys.argv[5])
