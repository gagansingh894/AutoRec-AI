const express = require('express');
const multer = require('multer');
const ejs = require('ejs');
const path = require('path');
const axios = require('axios');

// Set storage engine
const storage = multer.diskStorage({
    destination:'./public/uploads',
    filename: (req, file, cb) => {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
});


//  Init Upload
const upload = multer({
    storage: storage,
    limits: {filesize: 1000000},
    fileFilter: function(req, file, cb){
        checkFileType(file, cb)
    }
}).single('myImage');

function checkFileType(file, cb){
    const filetypes = /jpeg|jpg|png|gif/;
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = filetypes.test(file.mimetype);

    if (mimetype && extname){
        return cb(null, true);
    } else {
        cb('Error: Images Only!');
    }
}

const app = express();

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static('./public'));

app.get('/', (req, res) => res.render('home', {msg: ''}));


app.post('/upload', (req, res) => {
    upload(req, res, (err) => {
        console.log('started')
        if(err){
            res.render('home', {
                msg: err
            });
        } else {
                if(req.file == undefined){
                res.render('home', {
                    msg: 'Error: No File Selected!' 
                });
            } else {
                 axios.post('http://0.0.0.0:5001/detect').then(dres => {
                     if (dres.data.result == '-1') {
                        res.render('home', {
                            msg:'No Car Detected. Please upload a new image!'});
                        }
                    else {
                        console.log('all ok')
                        res.render('results', {
                        msg: dres.data.result[0].car_name,
                        rcmds: JSON.parse(dres.data.result[1].recommendations[0].result),
                        file:  `uploads/${req.file.filename}`
                            });

                    }
                }).catch(err => console.log(err));
    
                }                 

            }
        })
    })

app.listen(5000, () => console.log('Server started on port 5000'))