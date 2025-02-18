// Importación de módulos
var express = require('express')
var path = require('path')
var createError = require('http-errors')
var cors = require('cors')
var bodyParser = require('body-parser')
var app = express()

// Configuración de recursos estáticos
app.use(express.static(__dirname+'/'))

// Conexión a la base de datos MySQL
var dbMySQLNode = require('./database')

// Configuración del motor de plantillas y directorio de vistas
app.set('views', path.join(__dirname, '/'))
app.set('view engine', 'ejs')

// Configuración de análisis de solicitudes HTTP
app.use(bodyParser.json())
app.use(
  bodyParser.urlencoded({
    extended: true,
  }),
)

// Configuración de CORS
app.use(cors())

// Ruta principal
app.get('/', (req, res) => {
  res.render('index')
})

// Ruta para obtener datos desde la base de datos
app.get('/fetch', function (req, res) {
  dbMySQLNode.query('SELECT * FROM sensor ORDER BY id ', function (
    error,
    response,
  ) {
    if (error) {
      res.json({
        msg: error,
      })
    } else {
      res.json({
        msg: 'Data successfully fetched',
        sensor: response,
      })
    }
  })
})

// Inicio del servidor en el puerto 3000
app.listen(3000, function () {
  console.log('Node app is being served on port: 3000')
})
module.exports = app

