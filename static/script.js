function roadMode(){

fetch("/mode",{
method:"POST",
headers:{
"Content-Type":"application/x-www-form-urlencoded"
},
body:"mode=road"
})

alert("Road Mode Activated")

}

function marketMode(){

fetch("/mode",{
method:"POST",
headers:{
"Content-Type":"application/x-www-form-urlencoded"
},
body:"mode=market"
})

alert("Market Mode Activated")

}

function searchObject(){

let obj=document.getElementById("target").value

fetch("/target",{
method:"POST",
headers:{
"Content-Type":"application/x-www-form-urlencoded"
},
body:"target="+obj
})

alert("Searching for "+obj)

}
setInterval(function(){

    fetch("/state")
    .then(r=>r.text())
    .then(t=>{

        let text = ""

        if(t == "listening")
            text = "🎤 Listening..."

        else if(t == "speaking")
            text = "🔊 Speaking..."

        else
            text = "🟢 Ready"

        document.getElementById("stateBox").innerText = text

    })

},300)