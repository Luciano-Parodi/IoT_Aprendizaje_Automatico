var mysql = require('mysql')
var connection = mysql.createConnection({
  host: '127.0.0.1', 
  user: 'root',      
  password: 'X',    
  database: 'datos_sensor'

})
connection.connect((err) => {
  if (err) {
    console.log(err)
    return
  }
    /*connection.query("SELECT * FROM sensor", function (err, result, fields) {
      if (err) throw err;
      console.log(result);
    });*/
  
  console.log('Database connected')
})
module.exports = connection