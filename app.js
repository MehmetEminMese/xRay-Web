import express from 'express'
import dotenv from 'dotenv'
import conn from './db.js'
import pageRoute from "./routes/pageRoute.js"
dotenv.config()
conn()
const app=express()
const port =process.env.PORT
app.use(express.static('public'))
app.set('view engine','ejs')

app.use("/",pageRoute)

// app.get("/",(req,res)=>{
//   res.render('index')
// })
// app.get("/about",(req,res)=>{
//   res.render('about')
// })
// app.get("/foto",(req,res)=>{
//   res.render('foto')
// })
app.listen(port,()=>{
  console.log(`Application running on port:${port}`)
})